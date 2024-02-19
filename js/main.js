let progressBar = new ProgressBar();

//let corpus = new AudioCorpus(44100, 2, "corpusArea", progressBar);

//audio.loadFile("../corpus/theoshift.mp3");



async function recordAudio(audioCtx, opt) {
    // Step 1: Setup audio streamer
    let stream;
    let audioIOProcessor;
    try {
        stream = await navigator.mediaDevices.getUserMedia({audio:true});
    } catch (e) {
        console.log("Error opening audio: " + e);
    }
    try {
        await audioCtx.audioWorklet.addModule("audioioworklet.js");
        audioIOProcessor = new AudioWorkletNode(audioCtx, "audio-io-worklet");
    } catch(e) {
        console.log("Error loading particle worklet processor: " + e);
    }
    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(audioIOProcessor);
    audioIOProcessor.connect(audioCtx.destination);

    // Step 2: Setup particle worker and connect to audioIOProcessor
    let particleWorker = new Worker("particleworker.js");
    particleWorker.postMessage({"action":"initialize", "opt":opt});
    particleWorker.onmessage = function(event) {
        if (event.data.action == "postQuanta") {
            audioIOProcessor.port.postMessage({"action":"pushQuanta", "output":event.data.output})
        }
    }

    // Step 3: Finish setting up worker that processes particles, as well
    // as code to pass audio samples between that worker and the audio worklet
    audioIOProcessor.port.onmessage = function(event) {
        if (event.data.action == "inputQuanta") {
            particleWorker.postMessage({"action":"inputQuanta", "input":event.data.input});
        }
        else if (event.data.action == "pullQuanta") {
            // Audio processor is requesting a new output quantum
            particleWorker.postMessage({"action":"readQuanta"});
        }
    }
    




}


const audioCtx = new AudioContext();
recordAudio(audioCtx);



/*
navigator.mediaDevices
    .getUserMedia({audio:true})
    .then((stream) => {
        source = audioCtx.createMediaStreamSource(stream);
        const processor = new MyAudioProcessor();
        source.connect(processor);
        processor.connect(audioCtx.destination);
    })
    .catch(function(err) {
        console.log("Error: " + err);
    });
*/