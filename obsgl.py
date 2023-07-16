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

in int state[p];

uniform sampler2D WV;
uniform int t;

out float dot;

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
    
    
    // Step 2: Compute norm of approximation
    float norm = 0;
    for (int i = 0; i < M; i++) {
        float WHi = 0.0;
        for (int k = 0; k < p; k++) {
            WHi += h[k]*W(i, k);
        }
        norm += WHi*WHi;
    }
    norm = sqrt(norm);

    // Step 3: Compute the projection of the normed approximation
    // onto the observation
    dot = 0;
    for (int i = 0; i < M; i++) {
        float WHi = 0.0;
        for (int k = 0; k < p; k++) {
            WHi += h[k]*W(i, k);
        }
        float Vi = V(i, t);
        dot += Vi*WHi;
    }
    dot /= norm;
}
'''


class Observer:
    def __init__(self, p, W, V, L):
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
        WV = np.concatenate((W, V), axis=1)
        WV = np.array((WV.T).flatten(), dtype=np.float32)
        ctx = moderngl.create_standalone_context()
        texture = ctx.texture((M, N+T), 1, WV, dtype="f4")
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.use()
        program = ctx.program(
            vertex_shader=VERTEX_SHADER%(p, M, M, N, N, T, T, L),
            varyings=["dot"]
        )
        self.ut = program['t']
        self.ctx = ctx
        self.program = program

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
        Vi = np.zeros((Vt.shape[0], P))

        for i in range(P):
            ## Step 3: Apply observation update
            Wi = self.W[:, states[i]]
            hi = do_KL(Wi, Vt, self.L)
            Vi[:, i] = Wi.dot(hi).flatten()
        
        ViNorms = np.sqrt(np.sum(Vi**2, axis=0))
        ViNorms[ViNorms == 0] = 1
        Vi /= ViNorms[None, :]

        return np.sum(Vt[:, None]*Vi, axis=0)