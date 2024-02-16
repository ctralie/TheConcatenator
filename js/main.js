let progressBar = new ProgressBar();

//let corpus = new AudioCorpus(44100, 2, "corpusArea", progressBar);

//audio.loadFile("../corpus/theoshift.mp3");


async function recordAudio(audioCtx, hop) {
  let stream;
  let particleAudioProcessor;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
  } catch (e) {
    console.log("Error opening audio: " + e);
  }
  try {
    await audioCtx.audioWorklet.addModule("particleaudioworklet.js");
    particleAudioProcessor = new AudioWorkletNode(audioCtx, "particle-worklet-processor",
      {
        processorOptions: {
          hop: hop
        }
      }
    );

  } catch(e) {
    console.log("Error loading particle worklet processor: " + e);
  }
  const source = audioCtx.createMediaStreamSource(stream);
  source.connect(particleAudioProcessor);
  particleAudioProcessor.connect(audioCtx.destination);

}

//const audioCtx = new AudioContext();
//recordAudio(audioCtx, 1024);
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