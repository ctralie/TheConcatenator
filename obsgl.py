import moderngl
import struct
import numpy as np

VERTEX_SHADER = '''
#version 330
#define p   %i
#define M   %i
#define Mf  %i.0
#define N   %i
#define Nf  %i.0
#define T   %i
#define Tf  %i.0
#define L   %i
#define sigma %g

in int state[p];

uniform sampler2D WV;
uniform int t;
uniform float Vt_std;

out float prob;

float W(int i, int j) {
    return texture(WV, vec2((i+0.5)/Mf, (state[j]+0.5)/(Nf+Tf))).r;
}

float V(int i, int j) {
    return texture(WV, vec2((i+0.5)/Mf, (Nf+j+0.5)/(Nf+Tf))).r;
}

void main()
{
    float h[p];
    float facs[p];
    
    // Step 1: Perform KL iterations
    for (int j = 0; j < p; j++) {
        h[j] = 1;
    }
    for (int l = 0; l < L; l++) {
        // Determine the multiplicative factor for each component of h
        for (int j = 0; j < p; j++) {
            facs[j] = 0;
        }
        for (int i = 0; i < M; i++) {
            float WHi = 0.0;
            for (int k = 0; k < p; k++) {
                WHi += h[k]*W(i, k);
            }
            for (int j = 0; j < p; j++) {
                facs[j] += W(i, j)*V(i, t)/WHi;
            }
        }
        // Update each component of h
        for (int j = 0; j < p; j++) {
            h[j] *= facs[j];
        }
    }
    
    
    // Step 2: Compute observation probability
    prob = 0;
    for (int i = 0; i < M; i++) {
        float WHi = 0.0;
        for (int k = 0; k < p; k++) {
            WHi += h[k]*W(i, k);
        }
        float Vi = V(i, t);
        prob += Vi*log(Vi/WHi) - Vi + WHi;
    }
    prob = prob/(Vt_std*M);
    prob = exp(-prob*prob/(sigma*sigma));
}
'''


class Observer:
    def __init__(self, p, W, V, L, sigma):
        """
        Constructor for a class that computes observation probabilities
        quickly using moderngl

        Parameters
        ----------
        p: int
            Number of activations
        W: ndarray(M, N)
            Templates matrix, assumed to sum to 1 down the columns
        V: ndarray(M, T)
            Observations matrix
        L: int
            Number of iterations of KL
        sigma: float
            Observation noise
        """
        M = W.shape[0]
        N = W.shape[1]
        T = V.shape[1]
        self.p = p
        self.W = W
        self.V = V
        self.L = L
        self.sigma = sigma
        WV = np.concatenate((W, V), axis=1)
        WV = np.array((WV.T).flatten(), dtype=np.float32)
        ctx = moderngl.create_standalone_context()
        texture = ctx.texture((M, N+T), 1, WV, dtype="f4")
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.use()
        program = ctx.program(
            vertex_shader=VERTEX_SHADER%(p, M, M, N, N, T, T, L, sigma),
            varyings=["prob"]
        )
        self.ut = program['t']
        self.uVt_std = program['Vt_std']
        self.ctx = ctx
        self.program = program
        self.Vt_std = np.std(V, axis=0)

    def observe(self, states, t):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is the fast GPU version

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle
        t: int
            Time index
        
        Returns
        -------
        ndarray(P)
            Observation probabilities
        """
        P = states.shape[0]
        content = [
            (
            self.ctx.buffer(
                np.array(states.flatten(), dtype=np.int32)),
                '{}i'.format(self.p),
                'state'
            )
        ]
        vao = self.ctx.vertex_array(self.program, content)
        self.ut.value = t
        self.uVt_std.value = self.Vt_std[t]

        buffer = self.ctx.buffer(reserve=P*4)
        vao.transform(buffer, vertices=P)
        data = struct.unpack("{}f".format(P), buffer.read())
        data = np.array(data)
        return data
    
    
    def observe_slow(self, states, t):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is the slow CPU version

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle
        t: int
            Time index
        
        Returns
        -------
        ndarray(P)
            Observation probabilities
        """
        from probutils import do_KL
        P = states.shape[0]
        Vt = self.V[:, t]
        Vt_std = self.Vt_std[t]
        Vi = np.zeros((Vt.shape[0], P))

        for i in range(P):
            ## Step 3: Apply observation update
            Wi = self.W[:, states[i]]
            hi = do_KL(Wi, Vt, self.L)
            Vi[:, i] = Wi.dot(hi).flatten()
        
        Vt = Vt[:, None]
        num = (np.mean(Vt*np.log(Vt/Vi) - Vt + Vi, axis=0)/Vt_std)**2
        return np.exp(-num/self.sigma**2)