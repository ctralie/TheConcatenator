/**
 * Create audio samples in the wav format
 * @param {array} channels Array arrays of audio samples
 * @param {int} sr Sample rate
 */
function createWavURL(channels, sr) {
    const nChannels = channels.length;
    const N = channels[0].length;
    let audio = new Float32Array(N*nChannels);
    // Interleave audio channels
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < nChannels; j++) {
        audio[i*nChannels+j] = channels[j][i];
      }
    }
    // get WAV file bytes and audio params of your audio source
    const wavBytes = getWavBytes(audio.buffer, {
      isFloat: true,       // floating point or 16-bit integer
      numChannels: nChannels,
      sampleRate: sr,
    })
    const wav = new Blob([wavBytes], {type: 'audio/wav'});
    return window.URL.createObjectURL(wav);
}


class SampledAudio {
  constructor(audioContext, sr, nChannels) {
    this.audio = null;

    this.nChannels = nChannels;
    this.channels = [];
    this.audioContext = audioContext;
    this.sr = sr;
  }


  /**
   * Set the audio samples based on an array buffer
   * @param {ArrayBuffer} data Array buffer with audio data
   * @returns 
   */
  setSamplesAudioBuffer(data) {
    let that = this;
    return new Promise(resolve => {
      that.audioContext.decodeAudioData(data, function(buff) {
        that.channels = [];
        let c = buff.numberOfChannels;
        if (that.nChannels < c) {
          c = that.nChannels;
        }
        for (let i = 0; i < c; i++) {
          that.channels.push(buff.getChannelData(i));
        }
        // Pad up to the number of desired channels
        while (that.channels.length < that.nChannels) {
          that.channels.push(that.channels[0]);
        }
        that.sr = buff.sampleRate;
        that.src = createWavURL(that.channels, that.sr);
        resolve();
      });
    });
  }

  /**
   * Load in the samples from an audio file
   * @param {string} path Path to audio file
   * @returns A promise for when the samples have been loaded and set
   */
  loadFile(path) {
    let that = this;
    return new Promise((resolve, reject) => {
      $.get(path, function(data) {
        that.setSamplesAudioBuffer(data).then(resolve, reject);
      }, "arraybuffer")
      .fail(() => {
        reject();
      });
    });
  }

  /**
   * Put this audio into an audio player
   * @param {audio dom element} audioPlayer 
   */
  connectToPlayer(audioPlayer) {
    audioPlayer.src = this.src;
  }

  /**
   * Put this audio into an audio player and play it
   * @param {audio dom element} audioPlayer 
   */
  playAudio(audioPlayer) {
    audioPlayer.src = this.src;
    audioPlayer.play();
  }

  /**
   * Download the audio as a WAV
   */
  downloadAudio() {
    const a = document.createElement("a");
    a.href = createWavURL(this.channels, this.sr);
    a.style.display = 'none';
    a.download = "audio.wav";
    document.body.appendChild(a);
    a.click();
  }

}

class StreamingAudio {
  constructor(audioContext, sr, nChannels) {
    this.nChannels = nChannels;
    this.channels = [];
    this.audioContext = audioContext;
    this.sr = sr;
  }
}