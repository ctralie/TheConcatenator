#include <emscripten/bind.h>

using namespace emscripten;

#include <iostream>
#include <complex>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <ctime>
#include <math.h>



///////////////////////////////////////////////////////////////////////////////
//                            Fourier Code                                   //
///////////////////////////////////////////////////////////////////////////////
typedef std::complex<float> cfloat;
#define FFT_FORWARD 0
#define FFT_INVERSE 1
#define PI 3.1415926535897932384626433832795

float* getHannWindow(int N) {
	float* hannwindow = new float[N];
	for (int n = 0; n < N; n++) {
		float angle = 2.0*PI * n / (float)(N - 1);
		//Do a hann window for now
		hannwindow[n] = 0.54 - 0.46*cos(angle);
	}
	return hannwindow;
}

class DSP {
	private:
		cfloat*** W;//Cache the complex coefficients for the FFT
		float* hannwindow;
		int fftsize;

		/**
		 * Initialize the complex coefficients for the FFT
		 * @param fftsize The length of the fft
		 */
		void initCoeffs(int fftsize);

		/**
		 * Initialize the coefficients in Hann window
		 * @param fftsize The length of the fft
		 */
		void initWindow(int fftsize);
	
	public:
		cfloat fftres;
		DSP(int fftsize);
		~DSP();

		/**
		 * Implement the dft directly from the definition (used for speed comparison)
		 * @param sig Complex signal on which to compute dft
		 * @param toReturn The array that will hold the fourier coefficients
		 * @param N Length of signal
		 * @return Complex DFT coefficients
		 */
		void dft(cfloat* sig, cfloat* toReturn, int N);

		/**
		 * Perform an in-place Cooley-Tukey FFT
		 * @param toReturn Array that holds FFT coefficients
		 * @param N Length of array (assumed to be power of 2)
		 * @param inverse Whether this is a forward or inverse FFT
		 */
		void performfft(cfloat* toReturn, int N, int inverse);

		/**
		 * Perform the FFT on a complex signal
		 * @param sig The signal
		 * @param toReturn The array that will hold the fourier coefficients
		 * @param N Length of the signal (assumed to be power of 2)
		 * @return An N-length array with FFT coefficients
		 */
		void fft(cfloat* sig, cfloat* toReturn, int N);
	
		/**
		 * Perform the inverse FFT on an array of complex FFT coefficients
		 * @param sig The FFT coefficients
		 * @param toReturn The array that will hold the complex time series
		 * @param N Length of the FFT coefficients (assumed to be power of 2)
		 * @return An N-length array with FFT coefficients
		 */
		void ifft(cfloat* sig, cfloat* toReturn, int N);
		
		/**
		 * Helper function to create a complex array out of an array of 
		 * real amplitude samples
		 * @param data An array of floats for the audio data
		 * @param res Array holding the result
		 * @param N Total number of samples in data
		 * @param start Index to start in the array
		 * @param win Length of the window
		 * @param useWindow Whether to use the window
		 */
		void toWindowedComplexArray(float* data, cfloat* res, int N, int start, int win, bool useWindow);

		/**
		 * Perform a float-time fourier transform on a bunch of samples
		 * @param sig Samples in the signal
		 * @param N Length of signal
		 * @param win Window length
		 * @param hop Hop length
		 * @param useWindow Whether to use the window
		 * @param NWin Number of windows (returned by reference)
		 * @return An NWin x win 2D array of complex floats
		 */
		cfloat** stft(float* sig, int N, int win, int hop, bool useWindow, int* NWin);

		/**
		 * Perform a magnitude float-time fourier transform on a bunch of samples
		 * @param sig Samples in the signal
		 * @param N Length of signal
		 * @param win Window length
		 * @param hop Hop length
		 * @param useWindow Whether to use the window
		 * @param NWin Number of windows (returned by reference)
		 * @return A win x NWin 2D array of complex floats
		 */
		float** spectrogram(float* sig, int N, int win, int hop, int maxBin, bool useWindow, int* NWin);
};

/**
 * Free the memory associated to an STFT
 * @param S STFT
 * @param NWin Window length
 */
void deleteSTFT(cfloat** S, int NWin);

/**
 * Free the memory associated to a spectrogram
 * @param S Spectrogram
 * @param win Window length
 */
void deleteSpectrogram(float** S, int win);


/**
 * Compute the closest power of 2 greater than or equal
 * to some number
 * @param a The number
 * @return The closest power of 2 >= a
 */
int getClosestPowerOf2(int a) {
	float lg = log((float)a) / log(2.0);
	int power = (int)lg;
	if ((float)((int)lg) < lg) {
		power++;
	}
	return power;
}

/**
 * Compute a version of x which is bit-reversed
 * @param x A 32-bit int to reverse
 * @param length Length of bits
 * @return Bit-reversed version of x
 */
int bitReverse(int x, int length) {
	int toReturn = 0;
	int mirror = length / 2;
	for (int mask = 1; mask <= length; mask <<= 1, mirror >>= 1) {
		if ((mask & x) > 0)
			toReturn |= mirror;
	}
	return toReturn;
}


/**
 * Rearrange the terms in-place so that they're sorted by the least
 * significant bit (this is the order in which the terms are accessed
 * in the FFT)
 * @param a An array of complex numbers
 * @param N Number of elements in the array
 */
void rearrange(cfloat* a, int N) {
	for (int i = 0; i < N; i++) {
		int j = bitReverse(i, N);
		if (j > i) { //Don't waste effort swapping two mirrored
		//elements that have already been swapped
			cfloat temp = a[j];
			a[j] = a[i];
			a[i] = temp;
		}
	}
}

/**
 * Initialize the complex coefficients for the FFT
 * @param fftsize The length of the fft
 */
void DSP::initCoeffs(int fftsize) {
	int maxlevel = getClosestPowerOf2(fftsize) + 1;
	W = new cfloat**[maxlevel+1];
	for (int level = 1; level <= maxlevel; level++) {
		int FFTSize = 1 << level;
		W[level] = new cfloat*[2];
		W[level][0] = new cfloat[FFTSize >> 1];
		W[level][1] = new cfloat[FFTSize >> 1];
		for (int i = 0; i < FFTSize >> 1; i++) {
			float iangle = (float)i * 2.0 * PI / (float)FFTSize;
			float fangle = (-1.0) * iangle;
			W[level][FFT_FORWARD][i] = cfloat(cos(fangle), sin(fangle));
			W[level][FFT_INVERSE][i] = cfloat(cos(iangle), sin(iangle)); 
		}
	}
}

/**
 * Initialize the coefficients in Hann window
 * @param fftsize Lenght of fft
 */
void DSP::initWindow(int fftsize) {
	hannwindow = getHannWindow(fftsize);
}

/**
 * Perform an in-place Cooley-Tukey FFT
 * @param toReturn Array that holds FFT coefficients
 * @param N Length of array (assumed to be power of 2)
 * @param inverse Whether this is a forward or inverse FFT
 */
void DSP::performfft(cfloat* toReturn, int N, int inverse) {
	rearrange(toReturn, N);
	//Do the trivial FFT size of 2 first
	for (int i = 0; i < N; i += 2) {
		cfloat temp = toReturn[i];
		toReturn[i] = temp + toReturn[i + 1];
		toReturn[i + 1] = temp - toReturn[i + 1];
	}
	int Mindex = 2;//Index used to access the cached complex
	//coefficients
	for (int level = 2; level < N; level <<= 1) {
		int FFTSize = level << 1;
		for (int start = 0; start < N; start += FFTSize) {
			//This is a little chunk of an FFT of size "FFTSize"
			//to do in-place with the merging algorithm
			//NOTE: "level" gives the length between mirrored terms
			for (int i = 0; i < level; i++) {
				cfloat coeff = W[Mindex][inverse][i];
				cfloat first = toReturn[start + i];
				cfloat second = coeff*toReturn[start + i + level];
				toReturn[start + i] = first + second;
				toReturn[start + i + level] = first - second;
			}
		}
		Mindex++;
	}
}
	

DSP::DSP(int fftsize) {
	this->fftsize = fftsize;
	this->initCoeffs(fftsize);
	this->initWindow(fftsize);
}
DSP::~DSP() {
	int maxlevel = getClosestPowerOf2(fftsize) + 1;
	// Clean up FFT coefficients
	for (int level = 1; level <= maxlevel; level++) {
		for (int type = 0; type < 2; type++) {
			delete[] W[level][type];
		}
		delete[] W[level];
	}
	delete[] W;
	// Clean up window coefficients
	delete[] hannwindow;
}

/**
 * Implement the dft directly from the definition (used for speed comparison)
 * @param sig Complex signal on which to compute dft
 * @param toReturn The array that will hold the fourier coefficients
 * @param N Length of signal
 * @return Complex DFT coefficients
 */
void DSP::dft(cfloat* sig, cfloat* toReturn, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float angle = -2.0 * PI * (float)i * (float)j / (float)N;
			cfloat coeff(cos(angle), sin(angle));
			toReturn[i] = coeff*sig[i];
		}
	}
}

/**
 * Perform the FFT on a complex signal
 * @param sig The signal
 * @param toReturn The array that will hold the fourier coefficients
 * @param N Length of the signal (assumed to be power of 2)
 * @return An N-length array with FFT coefficients
 */
void DSP::fft(cfloat* sig, cfloat* toReturn, int N) {
	for (int i = 0; i < N; i++) {
		toReturn[i] = sig[i];
	}
	performfft(toReturn, N, FFT_FORWARD);	
}

/**
 * Perform the inverse FFT on an array of complex FFT coefficients
 * @param sig The FFT coefficients
 * @param toReturn The array that will hold the complex time series
 * @param N Length of the FFT coefficients (assumed to be power of 2)
 * @return An N-length array with FFT coefficients
 */
void DSP::ifft(cfloat* sig, cfloat* toReturn, int N) {
	for (int i = 0; i < N; i++) {
		toReturn[i] = sig[i];
		//Scale by 1/N for inverse FFT
		toReturn[i] *= cfloat(1.0/(float)N, 0);
	}
	performfft(toReturn, N, FFT_INVERSE);
}

/**
 * Helper function to create a complex array out of an array of 
 * real amplitude samples
 * @param data An array of floats for the audio data
 * @param res Array holding the result
 * @param N Total number of samples in data
 * @param start Index to start in the array
 * @param win Length of the window
 * @param useWindow Whether to use the window
 */
void DSP::toWindowedComplexArray(float* data, cfloat* res, int N, int start, int win, bool useWindow) {
	//Make a complex array out of the real array
	for (int i = 0; i < win; i++) {
		if (start+i < N) {
			res[i] = cfloat(data[start + i], 0.0);
			if (useWindow) {
				res[i] *= hannwindow[i];
			}
		}
		else {
			//Zero pad if not a power of 2
			res[i] = cfloat(0.0, 0.0);
		}
	}
}

/**
 * Perform a short-time fourier transform on a bunch of samples
 * @param sig Samples in the signal
 * @param N Length of signal
 * @param win Window length (Assumed to be a power of 2)
 * @param hop Hop length
 * @param useWindow Whether to use the window
 * @param NWin Number of windows (returned by reference)
 * @return An NWin x win 2D array of complex floats
 */
cfloat** DSP::stft(float* sig, int N, int win, int hop, bool useWindow, int* NWin) {
	*NWin = 1 + round((N-win)/(float)hop);
	cfloat** S = new cfloat*[*NWin];
	for (int i = 0; i < *NWin; i++) {
		S[i] = new cfloat[win];
	}
	cfloat* ffti = new cfloat[win];
	for (int i = 0; i < *NWin; i++) {
		toWindowedComplexArray(sig, S[i], N, i*hop, win, useWindow);
		fft(S[i], ffti, win);
		for (int j = 0; j < win; j++) {
			S[i][j] = ffti[j];
		}
	}
	delete[] ffti;
	return S;
}

/**
 * Perform a magnitude short-time fourier transform on a bunch of samples
 * @param sig Samples in the signal
 * @param N Length of signal
 * @param win Window length of STFT (Assumed to be a power of 2)
 * @param hop Hop length
 * @param useWindow Whether to use the window
 * @param NWin Number of windows (returned by reference)
 * @return A NWin x maxBin 2D array of floats
 */
float** DSP::spectrogram(float* sig, int N, int win, int hop, int maxBin, bool useWindow, int* NWin) {
	cfloat** SComplex = stft(sig, N, win, hop, useWindow, NWin);
	float** S = new float*[*NWin];
	for (int i = 0; i < *NWin; i++) {
		S[i] = new float[maxBin];
		for (int j = 0; j < maxBin; j++) {
			S[i][j] = abs(SComplex[i][j]);
		}
	}
	deleteSTFT(SComplex, *NWin);
	return S;
}

/**
 * Free the memory associated to an STFT
 * @param S Spectrogram
 * @param win Window length
 */
void deleteSTFT(cfloat** S, int NWin) {
	for (int i = 0; i < NWin; i++) {
		delete[] S[i];
	}
	delete[] S;
}



///////////////////////////////////////////////////////////////////////////////
//                      Particle Filter Code                                 //
///////////////////////////////////////////////////////////////////////////////


/**
 * A circular buffer for keeping track of the last activations to have been chosen
*/
class LastChosen {
	private:
		int pFinal;
		int r;
		int offset;

	public:
		int* idxs;

		/**
		 * @param  {int} pFinal: Sparsity parameter for final activations
		 * @param  {int} r: Repeated activations cutoff
		*/
		LastChosen(int pFinal, int r) {
			offset = 0;
			idxs = new int[pFinal*r];
			for (int i = 0; i < r*pFinal; i++) {
				idxs[i] = -1;
			}
		}
		
		/**
		 * @param{int} activations: New activations to push in
		*/
		void push(int* activations) {
			for (int j = 0; j < pFinal; j++) {
				idxs[offset*pFinal + j] = activations[j];
			}
			offset = (offset+1)%pFinal;
		}

		/**
		 * Return a pointer to the last window chosen (used to promote following states)
		*/
		int* getLastWindow() {
			return idxs + offset*pFinal;
		}

		~LastChosen() {
			delete[] idxs;
		}
};


/**
 * Encapsulate a mono circular buffer for reading in audio quantums
 * and processing hop lengths of audio at a time
*/
class AudioInBuffer {
	private:
		int win;
		DSP* dsp;
		float* hann;

		int quantum;
		float* buff;
		int offset; // Offset in samples of the circular shift
		std::mutex buffLock;
		cfloat* hannBuff; // Hann windowed copy of the buffer that's used


	public:
		/**
		 * @param{int} win: Window length (length of circular buffer)
		 * @param{int} quantum: Audio quantum. Should divide win/2, where we're assuming that the hop length is exactly half of the window length
		*/
		AudioInBuffer(int win, int quantum) {
			this->win = win;
			this->quantum = quantum;
			buff = new float[win];
			int offset = 0;
			hannBuff = new cfloat[win];
			dsp = new DSP(win);
			hann = getHannWindow(win);
		}

		/**
		 * Push a quantum of samples into the buffer
		 * 
		 * @param{float*} Pointer to quantum number of samples
		*/
		void pushQuantum(float* samples) {
			buffLock.lock();
			std::memcpy(buff+offset, samples, quantum*sizeof(float));
			offset = (offset + quantum) % win;
			buffLock.unlock();
		}

		/**
		 * Return true if enough quantums have been pushed to fill up
		 * hop samples, and which point an FFT is ready to go
		*/
		bool hopReady() {
			bool ret = 0;
			buffLock.lock();
			ret = offset % (win/2) == 0;
			buffLock.unlock();
			return ret;
		}

		/**
		 * Perform the FFT on the buffer in its current state
		 * 
		 * @param{float*} res: Array in which to store (kmax-kmin+1) magnitude frequencies
		 * @param{int} kmin: Minimum frequency bin to take in FFT
		 * @param{int} kmax: Maximum frequency bin to take in FFT 
		*/
		void performFFT(float* res, int kmin, int kmax) {
			if (!hopReady()) {
				std::cout << "Warning: Doing FFT before an entire hop chunk has been read in";
			}
			buffLock.lock();
			for (int i = 0; i < win; i++) {
				hannBuff[i] = cfloat(hann[i]*buff[(i+offset)%win]);
			}
			buffLock.unlock();
			dsp->performfft(hannBuff, win, FFT_FORWARD);
			for (int i = kmin; i <= kmax; i++) {
				res[i-kmin] = abs(hannBuff[i]);
			}
		}

		/**
		 * This method is for debugging; we usually only need the FFT
		 * of samples
		 * res: Array in which to store win samples
		*/
		void copyWindow(float* res) {
			buffLock.lock();
			for (int i = 0; i < win; i++) {
				res[i] = hann[i]*buff[(i+offset)%win];
			}
			buffLock.unlock();
		}

		~AudioInBuffer() {
			delete[] hann;
			delete dsp;
			delete[] buff;
			delete[] hannBuff;
		}
};

/*
* Store a buffer of increasing size for the audio that's being outputted.
* Double the buffer size if we run out of space
* Keep track of an index at which audio quantums are being read out, as well
* as an index at which new window are to be placed
*/
class AudioOutBuffer {
	private:
		int N; // Current size of buffer
		int quantum; // Quantum size
		int q; // Location of quantum
		int win; // Window length
		int w; // Pointer to current window location

		float* samples;
		std::mutex samplesLock;
		float* hann;

		void doubleCapacity() {
			samplesLock.lock();
			float* newSamples = new float[N*2];
			delete[] samples;
			samples = newSamples;
			N *= 2;
			samplesLock.unlock();
		}

	public:
		AudioOutBuffer(int win, int sr, int quantum) {
			N = sr*60; // Start out with 60 seconds worth of audio in the buffer
			samples = new float[N];
			this->quantum = quantum;
			q = 0;
			this->win = win;
			w = win/2; // Start off with a latency of hop, so output all 0's for the first hop (TODO: Can play with this)
			hann = getHannWindow(win);
		}

		/**
		 * Add a new window of samples on top of what's already there,
		 * applying a Hann window
		 * 
		 * @param{float*} newSamples: New window to be added in
		*/
		void addWindow(float* newSamples) {
			if (w + win > N) {
				doubleCapacity();
			}
			samplesLock.lock();
			for (int i = 0; i < win; i++) {
				samples[w+i] += hann[i]*newSamples[i];
			}
			//std::cout << "samples[" << win*3/2 << "] = " << samples[win*3/2] << "\n";
			w += win/2; // Shift over by a hop
			samplesLock.unlock();
		}

		/**
		 * Copy the next quantum of audio to an out location
		*/
		void readNextQuantum(float* out) {
			samplesLock.lock();
			memcpy(out, samples+q, quantum);
			//std::cout << "q:" << q << ", w: " << w << ", samples[" << q << "]: " << out[0];
			q += quantum;
			if (q > w) {
				std::cout << "Warning: buffer not filling quickly enough to output audio quantum";
			}
			samplesLock.unlock();
		}

		~AudioOutBuffer() {
			delete[] samples;
			delete[] hann;
		}
};

class ParticleFilter {
	private:
		// Particle and particle parameters
		int* particles;
		float* weights;
		LastChosen* lastChosen;// Circular buffer of last choices
		int p;
		int pFinal;
		int P;
		float pd;
		float temperature;
		int L;
		int r;
		float neffThresh;

		// Audio/feature buffers
		int win;
		int quantum;
		int sr;
		int nChannels;
		AudioInBuffer**  inBuffers;
		AudioOutBuffer** outBuffers;
		float* Vt;
		float* quantaIn;
		float* quantaOut;


	void initializeRandomParticles() {
		// TODO: Finish this; use proper corpus size
		std::srand(std::time(NULL));
		particles = new int[P*p];
		particlesShare = (uintptr_t)particles;
		for (int i = 0; i < P*p; i++) {
			particles[i] = std::rand() / ((RAND_MAX + 1u) / 100);
		}
	}

	public:
		int kmin;
		int kmax;

		// Make public pointers to shared buffers
		uintptr_t particlesShare;
		uintptr_t VtShare;
		uintptr_t quantaInShare;
		uintptr_t quantaOutShare;
		
		/**
		 * ParticleFilter constructor
		 * 
		 * @param  {int} win           : Window length for each STFT window.  For simplicity, assume that hop is 1/2 of this
		 * @param  {int} quantum       : Audio quantum. Should divide win/2, where we're assuming that the hop length is exactly half of the window length
		 * @param  {int} sr            : Audio sample rate
		 * @param  {int} nChannels     : Number of audio channels
		 * @param  {float} minFreq     : Minimum frequency to use (in hz)
		 * @param  {float} maxFreq     : Maximum frequency to use (in hz)
		 * @param  {int} p             : Sparsity parameter for particles
		 * @param  {int} pFinal        : Sparsity parameter for final activations
		 * @param  {int} P             : Number of particles
		 * @param  {float} pd          : State transition probability
		 * @param  {float} temperature : Amount to focus on matching observations
		 * @param  {int} L             : Number of iterations for NMF observation probabilities
		 * @param  {int} r             : Repeated activations cutoff
		 */
		ParticleFilter(int win, int quantum, int sr, int nChannels, float minFreq, float maxFreq, int p, int pFinal, int P, float pd, float temperature, int L, int r) {
			this->win = win;
			this->quantum = quantum;
			this->sr = sr;
			this->p = p;
			this->pFinal = pFinal;
			this->P = P;
			this->pd = pd;
			this->temperature = temperature;
			this->L = L;
			this->r = r;
			this->nChannels = nChannels;
			neffThresh = 0.1*((float)P);
			
			// Step 1: Setup buffers and windows
			kmin = (int)(win*minFreq/((float)sr)) + 1;
			kmax = (int)(win*maxFreq/((float)sr)) + 1;
			if (kmax > win/2) {
				kmax = win/2;
			}
			lastChosen = new LastChosen(pFinal, r);
			this->nChannels = nChannels;
			inBuffers =  new AudioInBuffer*[nChannels];
			outBuffers = new AudioOutBuffer*[nChannels];
			for (int i = 0; i < nChannels; i++) {
				inBuffers[i] =  new AudioInBuffer(win, quantum);
				outBuffers[i] = new AudioOutBuffer(win, sr, quantum);
			}
			Vt = new float[(kmax-kmin+1)*nChannels];
			VtShare = (uintptr_t)Vt;
			quantaIn = new float[quantum*nChannels];
			quantaInShare = (uintptr_t)quantaIn;
			quantaOut = new float[quantum*nChannels];
			quantaOutShare = (uintptr_t)quantaOut;

			// Step 2: Allocate particles randomly
			this->initializeRandomParticles();
			
			// TODO: Finish this

		}

		/**
		 * Add an audio quantum to the input buffer of each channel
		 * 
		 * @param{uintptr_t} psamples: A pointer to the audio samples to incorporate,
		 * 						      where the channels are stored back to back
		 * 
		 * @return True if the next hop samples are ready
		*/
		bool inputNextQuanta(uintptr_t psamples) {
			float* samples = (float*)psamples;
			//std::cout << "inputNextQuanta samples[0]: " << samples[0] << "\n";
			for (int ch = 0; ch < nChannels; ch++) {
				inBuffers[ch]->pushQuantum(samples + ch*quantum);
			}
			return inBuffers[0]->hopReady();
		}

		/**
		 * Read out the next quantum of audio samples for each channel, concatenated
		 * one after another channel-wise in memory
		 * 
		 * @param{uintptr_t} psamples: A pointer to where the samples should be written
		*/
		void readNextQuanta(uintptr_t psamples) {
			float* samples = (float*)psamples;
			for (int ch = 0; ch < nChannels; ch++) {
				outBuffers[ch]->readNextQuantum(samples + ch*quantum);
			}
			//std::cout << "readNextQuanta samples[0]: " << samples[0] << "\n";
		}

		// TODO: Fill in methods that do different steps of the particle filter
		
		/**
		 * This method is for debugging; it will be replaced by a method that performs
		 * a particle filter
		*/
		void addWindows() {
			// Do the FFT just to test timing
			float* res = new float[win];
			//std::cout << "Doing fft\n";
			for (int ch = 0; ch < nChannels; ch++) {
				inBuffers[ch]->performFFT(res, kmin, kmax);
				inBuffers[ch]->copyWindow(res);
				outBuffers[ch]->addWindow(res);
			}
			delete[] res;
		}

		void printHello() {
			std::cout << "Hello!\n";
		}



		~ParticleFilter() {
			delete lastChosen;
			delete[] particles;
			for (int i = 0; i < nChannels; i++) {
				delete inBuffers[i];
				delete outBuffers[i];
			}
			delete[] inBuffers;
			delete[] outBuffers;
			delete[] Vt;
			delete[] quantaIn;
			delete[] quantaOut;
		}
};

EMSCRIPTEN_BINDINGS(my_module) {
    class_<ParticleFilter>("ParticleFilter")
		.constructor<int, int, int, int, float, float, int, int, int, float, float, int, int>()
		.function("inputNextQuanta", &ParticleFilter::inputNextQuanta, allow_raw_pointers())
		.function("readNextQuanta", &ParticleFilter::readNextQuanta, allow_raw_pointers())
		.function("addWindows", &ParticleFilter::addWindows)
		.function("printHello", &ParticleFilter::printHello)
		.property("particlesShare", &ParticleFilter::particlesShare)
		.property("VtShare", &ParticleFilter::VtShare)
		.property("quantaInShare", &ParticleFilter::quantaInShare)
		.property("quantaOutShare", &ParticleFilter::quantaOutShare)
		.property("kmin", &ParticleFilter::kmin)
		.property("kmax", &ParticleFilter::kmax)
		;
}