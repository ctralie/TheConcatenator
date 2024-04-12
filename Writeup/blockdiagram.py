import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.append("../src")
from audioutils import *

n_per_color = 6
colors = ["Blues", "Oranges", "Greens", "Purples"]
x = np.linspace(0.2, 0.8, n_per_color)
rgb = []
for c in colors:
    rgb.append(plt.colormaps.get_cmap(plt.get_cmap(c))(x)[:, 0:3])
rgb = np.concatenate(rgb, axis=0)

def draw_grid(rgb):
    N = rgb.shape[0]
    plt.yticks(np.arange(N), [""]*N)
    plt.xticks(np.arange(N), [""]*N)
    for i in range(N):
        for j in range(N):
            plt.plot([j, j], [0, i], c=rgb[j, :], linewidth=1, linestyle=':', zorder=0)
            plt.plot([0, j], [i, i], c=rgb[i, :], linewidth=1, linestyle=':', zorder=0)
    plt.xlim([-0.5, N-0.5])
    plt.ylim([-0.5, N-0.5])

def draw_color_key(rgb):
    ax = plt.gca()
    N = rgb.shape[0]
    for i in range(N):
        start = i-0.2
        if i > 0:
            start -= 0.3
        r = Rectangle((-1, start), 1, 1, facecolor=rgb[i, :])
        ax.add_patch(r)
        r = Rectangle((start, -1), 1, 1, facecolor=rgb[i, :])
        ax.add_patch(r)
    plt.xlim([-1, N])
    plt.ylim([-1, N])
    plt.xlabel("$\\vec{s_i}[0]$")
    plt.ylabel("$\\vec{s_i}[1]$")

def plot_states(states, w, rgb, alpha=1):
    c = np.zeros((states.shape[0], 4))
    c[:, 0:3] = rgb[states[:, 0]]
    c[:, -1] = alpha
    plt.scatter(states[:, 0], states[:, 1], s=w*4, c=c)
    c[:, 0:3] = rgb[states[:, 1]]
    plt.scatter(states[:, 0], states[:, 1], s=w, c=c)

N = rgb.shape[0]
P = 50

fac = 0.65
plt.figure(figsize=(fac*30*4/5, fac*5))

seed = 1

plt.clf()
np.random.seed(seed)
states = np.random.randint(n_per_color-1, size=(P, 2))
states += np.random.randint(len(colors), size=(P, 2))*n_per_color
states[states == 0] = 1
states = set([(x, y) for [x, y] in states])
states = np.array(list(states))
P = states.shape[0]
print("P = ", P)

w = 50*np.random.rand(P)[::-1]
w[5] *= 2

states2 = np.array(states)+1
idx = np.random.rand(*states2.shape) > 0.9
print(np.sum(idx))
states2[idx] = np.random.randint(1, N, size=np.sum(idx))
diff = states2 - states


colspan = 5
sdim = (3, 4*colspan)

## Step 1: Plot initial states
plt.subplot2grid(sdim, (0, 0), rowspan=3, colspan=colspan)
plot_states(states, w, rgb)
draw_color_key(rgb)
plt.title("Beginning of Timestep")

## Step 2: Plot states after transitions
plt.subplot2grid(sdim, (0, 1*colspan), rowspan=3, colspan=colspan)
plot_states(states, w, rgb, alpha=0.1)
plot_states(states2, w, rgb)
hl = 0.7 # Head length
for i in range(states.shape[0]):
    [x, y] = states[i]
    diff = states2[i] - states[i]
    mag = np.sqrt(np.sum(diff**2))
    diff = ((mag-hl)/mag)*diff
    [u, v] = diff
    plt.arrow(x, y, u, v, head_width=hl/2, head_length=hl/2, zorder=105, facecolor='k', edgecolor='k', width=0.01)
draw_color_key(rgb)
plt.yticks([])
plt.ylabel("")
plt.title("1. Apply Transition Model ($p_d$)")

## Step 3: Plot states after observation probabilities
plt.subplot2grid(sdim, (0, 2*colspan), rowspan=3, colspan=colspan)
idx = np.zeros_like(w)
promote = [6, 7]
for j in range(2):
    for iup in promote:
        idx += np.array(states2[:, j] == iup, dtype=float)
print(np.max(idx))
w2 = np.array(w)
w2 *= (0.25 + 2*idx)
print("Num upweighted", np.sum(idx > 0))
idx = np.argsort(-w2)[0:5]
plt.scatter(states2[idx, 0], states2[idx, 1], s=1000, facecolor='none', edgecolor='k', linestyle='--')
plot_states(states2, w2, rgb)
draw_color_key(rgb)
plt.yticks([])
plt.ylabel("")
plt.title("2. Apply Observation Weights ($\\tau$)")

## Step 4a: Show aggregated weights
plt.subplot2grid(sdim, (2, 3*colspan), rowspan=1, colspan=colspan)
agg = np.zeros(N)
for i in idx:
    agg[states2[i]] += w2[i]
ax = plt.gca()
for i, a in enumerate(agg):
    r = Rectangle((i, 0), 1, a+2, facecolor=rgb[i, :])
    ax.add_patch(r)
idx = np.argsort(-agg)[0:2]
plt.scatter(idx+0.5, agg[idx], c='k', marker='x', s=80)
plt.xlim([0, agg.size])
plt.ylim([0, np.max(agg)+10])
plt.axis("off")
plt.title("3a. Aggregate Top 0.1P Particles")


## Step 4b: Show activations and audio added together
sr = 44100
win = 2048
hop = win//2
ycorpus = load_corpus("../corpus/EdenVIP2", sr, False)
WSi, WPi = get_windowed(ycorpus[0, :], win, lambda N: np.ones(N))
alpha = 0.2
beta = 0.8
x = WSi[:, 1051]
x /= np.max(np.abs(x))
y = WSi[:, 807]
y /= np.max(np.abs(y))
x *= alpha
y *= beta

plt.subplot2grid(sdim, (1, 3*colspan), rowspan=1, colspan=1)
ax = plt.gca()
r = Rectangle((0, 0), 1, alpha, facecolor=rgb[idx[0], :])
ax.add_patch(r)
r = Rectangle((1, 0), 1, beta, facecolor=rgb[idx[1], :])
ax.add_patch(r)
plt.xlim([0, 2])
plt.ylim([0, 1])
plt.axis("off")
plt.subplot2grid(sdim, (1, 3*colspan+1), rowspan=1, colspan=colspan-1)
plt.plot(x, c=rgb[idx[0], :])
plt.plot(y, c=rgb[idx[1], :])
plt.axis("off")
plt.title("3b. Choose Activations + Apply KL")


## Step 5: Show superimposed audio
winfn = hann_window(win)
z = (x+y)*winfn
i1 = int(sr*10.25)
wlen = win*2
all_audio = ycorpus[0, i1:i1+wlen]
all_audio[-hop:] *= winfn[-hop:]
sidx = all_audio.size-win+hop

waveform_lw = 1
plt.subplot2grid(sdim, (0, 3*colspan), rowspan=1, colspan=colspan)
plt.plot(all_audio, c='k', linewidth=waveform_lw)
plt.plot(np.arange(win)+wlen-hop, z, c=alpha*rgb[idx[0], :]+beta*rgb[idx[1], :])
plt.plot([sidx, sidx], [-1, 1], c='k', linestyle='--')
plt.plot([sidx+hop, sidx+hop], [-1, 1], c='k', linestyle='--')
plt.plot([sidx, sidx+hop], [-1, -1], c='k', linestyle='--')
plt.plot([sidx, sidx+hop], [1, 1], c='k', linestyle='--')
plt.title("3c. Window And Mix")

plt.axis("off")
plt.subplot2grid(sdim, (1, 3*colspan), rowspan=1, colspan=colspan//2)
all_audio = np.concatenate((all_audio, np.zeros(hop)))
all_audio[-win:] += z
plt.plot(all_audio[sidx:sidx+hop], c='k', linewidth=waveform_lw)
plt.axis("off")

plt.savefig("BlockDgm.svg", bbox_inches='tight')
