import moderngl
import struct
import numpy as np

TEXTURE_WIDTH = 1024

VERTEX_SHADER = '''
#version 330
#define p   %i
#define M   %i
#define Mf  %i.0
#define L   %i
#define TEXTURE_WIDTH %i
#define TEXTURE_WIDTH_F %i.0

in int state[p];

uniform sampler2DArray WTex;
uniform sampler2D VTex;

out float dot;

float W(int i, int j) {
    int jj = state[j] %% TEXTURE_WIDTH;
    int layer = state[j] / TEXTURE_WIDTH;
    return texture(WTex, vec3((i+0.5)/Mf, (jj+0.5)/TEXTURE_WIDTH_F, layer)).r;
}

float V(int i) {
    return texture(VTex, vec2(0.5, (i+0.5)/Mf)).r;
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
                facs[j] += W(i, j)*V(i)/WHi;
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
        dot += V(i)*WHi;
    }
    dot /= norm;
}
'''


class Observer:
    def __init__(self, p, W, L):
        """
        Constructor for a class that computes observation probabilities
        quickly using moderngl

        Parameters
        ----------
        p: int
            Number of activations
        W: ndarray(M, N)
            Templates matrix, assumed to sum to 1 down the columns
        L: int
            Number of iterations of KL
        sigma: float
            Observation noise
        """
        ctx = moderngl.create_standalone_context()

        M = W.shape[0]
        N = W.shape[1]
        self.p = p
        self.W = W
        self.L = L

        # Break up templates into a stack of textures
        n_stack = int(np.ceil(N/TEXTURE_WIDTH))
        WTex = np.zeros((n_stack, TEXTURE_WIDTH, M), dtype=np.float32)
        for i in range(n_stack):
            Wi = W[:, i*TEXTURE_WIDTH:(i+1)*TEXTURE_WIDTH].T
            WTex[i, 0:Wi.shape[0], :] = Wi
        texture = ctx.texture_array((M, TEXTURE_WIDTH, n_stack), 1, WTex, dtype="f4")
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.use(0)

        program = ctx.program(
            vertex_shader=VERTEX_SHADER%(p, M, M, L, TEXTURE_WIDTH, TEXTURE_WIDTH),
            varyings=["dot"]
        )
        program['WTex'].value = 0
        program['VTex'].value = 1
        self.ctx = ctx
        self.program = program

    def observe(self, states, Vt):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is the fast GPU version

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle
        Vt: ndarray(M)
            Observation for this time
        
        Returns
        -------
        ndarray(P)
            Observation probabilities
        """
        VTex = np.array(Vt[:, None], dtype=np.float32)
        VTex = self.ctx.texture((1, Vt.size), 1, VTex, dtype="f4")
        VTex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        VTex.use(1)

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

        buffer = self.ctx.buffer(reserve=P*4)
        vao.transform(buffer, vertices=P)
        data = struct.unpack("{}f".format(P), buffer.read())
        data = np.array(data)
        return data
    
    
    def observe_cpu(self, states, t):
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