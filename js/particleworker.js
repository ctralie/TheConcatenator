/**
 * Used to carry out expensive particle filter tasks in a thread
 * 
 */
importScripts("particle.js");

const DEFAULT_PARTICLE_OPTIONS = 
{
    quantum:128,
    win:2048,
    sr:44100,
    nChannels:1,
    minFreq:50,
    maxFreq:10000,
    p:10,
    pFinal:10,
    P:2000,
    temperature:100,
    L:10,
    r:7,
    pd:0.99
};


let opt = {};
let moduleLoaded = new Promise((resolve) => {
    Module.onRuntimeInitialized = () => {resolve();};
});


class ParticleFilter {
    constructor() {
        this.initialized = false;
    }

    /**
     * Initialize the particle filter and setup the web assembly module
     * with pointers to shared buffers
     * 
     * @param {object} event Event passed along to onmessage, which can specify
     *                       a field "data.opts". Otherwise, defaults are used
     */
    initialize(event) {
        const that = this;
        moduleLoaded.then(() => {
            // Step 1: Setup parameters
            opt = event.data.opt;
            if (opt === undefined) {
                opt = {};
            }
            for (let key in DEFAULT_PARTICLE_OPTIONS) {
                if (!(key in opt)) {
                    opt[key] = DEFAULT_PARTICLE_OPTIONS[key];
                }
                that[key] = opt[key];
            }
            // Step 2: Setup particle filter object with pointers to heap
            let m = new Module.ParticleFilter(that.win, that.quantum, that.sr, that.nChannels, that.minFreq, that.maxFreq, that.p, that.pFinal, that.P, that.pd, that.temperature, that.L, that.r);
            that.module = m;
            that.particles = Module.HEAP32.subarray(m.particlesShare >> 2);
            that.Vt = Module.HEAPF32.subarray(m.VtShare >> 2);
            that.quantaIn = [];
            that.quantaOut = [];
            for (let ch = 0; ch < that.nChannels; ch++) {
                that.quantaIn[ch] = Module.HEAPF32.subarray((m.quantaInShare >> 2) + ch*that.quantum);
                that.quantaOut[ch] = Module.HEAPF32.subarray((m.quantaOutShare >> 2) + ch*that.quantum);
            }
            that.initialized = true;
        });
    }

    inputQuanta(input) {
        if (this.initialized) {
            for (let ch = 0; ch < this.nChannels; ch++) {
                this.quantaIn[0].set(input[ch%input.length]);
            }
            const hopReady = this.module.inputNextQuanta(this.module.quantaInShare);
            if (hopReady) {
                this.module.addWindows();
            }
        }
        else {
            console.log("Warning: trying to input audio quanta before particle filter object is instantiated");
        }
    }

    readQuanta() {
        if (this.initialized) {
            this.module.readNextQuanta(this.module.quantaOutShare);
            const quantum = this.quantum;
            let output = [];
            // TODO: It's annoying that I have to copy over audio sample by sample
            // because I couldn't get the buffer copying to work, but this does not
            // seem to be a bottleneck
            for (let i = 0; i < this.nChannels; i++) {
                output.push(new Float32Array(quantum));
                for (let k = 0; k < quantum; k++) {
                    output[i][k] = this.quantaOut[i][k];
                }
            }
            postMessage({"action":"postQuanta", "output":output});
        }
    }
}


let pf = new ParticleFilter();
onmessage = function(event) {
    if (event.data.action == "initialize") {
        pf.initialize(event);
    }
    if (event.data.action == "inputQuanta") {
        pf.inputQuanta(event.data.input);
    }
    if (event.data.action == "readQuanta") {
        pf.readQuanta();
    }
}