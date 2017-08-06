#include "OptionPricing.h"

// Kernel functions

__global__ void InitializeMuKernel(float * MuIto, const float r, const float * q, const float * Sigma, const float * S0, const int nStocks, const float T) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < nStocks) {
		MuIto[i] = S0[i] * __expf((r - q[i] - (0.5f * Sigma[i] * Sigma[i])) * T);
	}
}

__global__ void GeneratePathKernel(float * __restrict__ Stock, const float * __restrict__ MuIto, const float * __restrict__ Sigma, const int nStocks, const int Iter) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Iter) {
		for (int j = 0, k = i; j < nStocks; ++j, k += Iter) {
			Stock[k] = MuIto[j] * __expf(Sigma[j] * Stock[k]);
		}
	}
}

__global__ void OnMax(float * __restrict__ P, const float * __restrict__ StockPaths, const int nStocks, const int Iter) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Iter) {
		P[i] = StockPaths[i];
		for (int j = 1, k = i; j < nStocks; ++j, k += Iter){
			P[i] = max(P[i], StockPaths[k]);
		}
	}
}

__global__ void OnMin(float * __restrict__ P, const float * __restrict__ StockPaths, const int nStocks, const int Iter) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Iter) {
		P[i] = StockPaths[i];
		for (int j = 1, k = i; j < nStocks; ++j, k += Iter){
			P[i] = min(P[i], StockPaths[k]);
		}
	}
}

__global__ void Call(float * P, const float K, const float Discount, const int Size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Size) {
		P[i] = max(P[i] - K, 0.f);
	}
}

__global__ void Put(float * P, const float K, const float Discount, const int Size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Size) {
		P[i] = max(K - P[i], 0.f);
	}
}

__global__ void Best(float * P, const float K, const float Discount, const int Size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < Size) {
		P[i] = max(P[i], K);
	}
}

__global__ void MakeLower(float * X, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		for (int j = i + 1; j < n; ++j)
			X[(j * n) + i] = 0.f;
	}
}

// Utility functions

inline float * cudaAlloc(size_t n) {
	// Wrapper around cudaMalloc()
	float* Ptr = 0;
	cudaMalloc(&Ptr, n * sizeof(float));
	if (!Ptr)
		throw std::bad_alloc();
	return Ptr;
}

void CholeskyFactor(float * Matrix, const int n) {
	// Stores the Cholesky decomposition of the matrix in-place. Overwrites the lower part of Matrix
	int work_size = 0;
	int * unused = nullptr;
	cusolverDnHandle_t solver_handle;

	cusolverDnCreate(&solver_handle);
	cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, n, Matrix, n, &work_size);
	auto work = cudaAlloc(work_size);
	cudaMalloc(&unused, sizeof(int));
	cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, n, Matrix, n, work, work_size, unused);
	cudaFree(work);
	cudaFree(unused);
	cusolverDnDestroy(solver_handle);
	MakeLower<<<(n + TPB - 1)/ TPB, TPB>>>(Matrix, n);
}

inline void ApplyCorrelation(const float * __restrict__ Wsrc, float * __restrict__ Wdst, float * __restrict__ A, const int Iter, const int nStocks) {
	// W <- A * W where A*At = Correlation matrix
	cublasSgemm('n', 't', Iter, nStocks, nStocks, 1.0f, Wsrc, Iter, A, nStocks, 0.0f, Wdst, Iter);
}

void RandomFill(float * Vect, const int n, const float T) {
	// Initialize PRNG
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(gen, std::random_device{}());

	// Fill with iid N(0,T) variables knowing that W(T) = N(0, T)
	curandGenerateNormal(gen, Vect, n, 0.f, sqrtf(T));

	// Destroy PRNG.
	curandDestroyGenerator(gen);
}

inline void GeneratePath(float * __restrict__ Stock, const float * __restrict__ MuIto, const float * __restrict__ Sigma, const int nStocks, const int Iter) {
	// S[i, j] = S0[i] * exp((r - q[i] - Sigma[i]^2 / 2)*T + Sigma[i]*Wij(T))
	//      = MuIto[i] * exp(Sigma[i] * W[i,j])
	auto _TPB = 1024;
	GeneratePathKernel<<<(Iter + _TPB - 1) / _TPB, _TPB>>>(Stock, MuIto, Sigma, nStocks, Iter);
}

inline void InitializeMu(float * MuIto, const float r, const float * q, const float * Sigma, const float * S0, const int nStocks, const float T) {
	// MuIto[i] = S0[i] * exp(((Sigma[i] * Sigma[i]) * (-0.5) + r - q[i]) * T)
	InitializeMuKernel<<<(nStocks + TPB - 1) / TPB, TPB>>>(MuIto, r, q, Sigma, S0, nStocks, T);
}

// Payoffs Implementation

inline void CallOnMax(float * P, const float * StockPaths, const float K, const float Discount, const int nStocks, const int Iter) {
	OnMax<<<(Iter + TPB - 1) / TPB, TPB>>>(P, StockPaths, nStocks, Iter);
	Call<<<(Iter + TPB - 1) / TPB, TPB>>>(P, K, Discount, Iter);
}

inline void CallOnMin(float * P, const float * StockPaths, const float K, const float Discount, const int nStocks, const int Iter) {
	OnMin<<<(Iter + TPB - 1) / TPB, TPB>>>(P, StockPaths, nStocks, Iter);
	Call<<<(Iter + TPB - 1) / TPB, TPB>>>(P, K, Discount, Iter);
}

inline void PutOnMax(float * P, const float * StockPaths, const float K, const float Discount, const int nStocks, const int Iter) {
	OnMax<<<(Iter + TPB - 1) / TPB, TPB>>>(P, StockPaths, nStocks, Iter);
	Put<<<(Iter + TPB - 1) / TPB, TPB>>>(P, K, Discount, Iter);
}

inline void PutOnMin(float * P, const float * StockPaths, const float K, const float Discount, const int nStocks, const int Iter) {
	OnMin<<<(Iter + TPB - 1) / TPB, TPB>>>(P, StockPaths, nStocks, Iter);
	Put<<<(Iter + TPB - 1) / TPB, TPB>>>(P, K, Discount, Iter);
}

inline void BestOfAssetsOrCash(float * P, const float * StockPaths, const float K, const float Discount, const int nStocks, const int Iter) {
	OnMax<<<(Iter + TPB - 1) / TPB, TPB>>>(P, StockPaths, nStocks, Iter);
	Best<<<(Iter + TPB - 1) / TPB, TPB>>>(P, K, Discount, Iter);
}

// Parser Implementation

inline bool loadFile(const char * file_name, std::string& file_contents, const char sep) {
	FILE * file_handle = nullptr;
	fopen_s(&file_handle, file_name, "r");
	if (!file_handle)
		return false;
	fseek(file_handle, 0L, SEEK_END);
	auto file_size = ftell(file_handle);
	file_contents.resize(file_size + 1);
	rewind(file_handle);
	fread(&file_contents[0], sizeof(char), file_size, file_handle);
	// Parse separators 
	std::transform(file_contents.begin(), file_contents.end(), file_contents.begin(), [&](char& c) -> char {
		return (c != sep && c != '\n') ? c : ' ';
	});
	file_contents[file_size] = '\0';
	fclose(file_handle);
	return true;
}

void fillFloats(float * float_array, const int element_count, std::string& file_contents, const char * locale_string) {
	std::istringstream stream(file_contents);
	if (locale_string)
		stream.imbue(std::locale(locale_string));
	for (int i = 0; i < element_count; ++i) {
		stream >> float_array[i];
	}
}

void parseFile(float * device_float_array, const char * file_name, const int element_count, const char sep = ',', const char * locale_string = nullptr) {
	std::string Contents;
	auto Array = new float[element_count];
	loadFile(file_name, Contents, sep);
	fillFloats(Array, element_count, Contents, locale_string);
	cudaMemcpy(device_float_array, Array, element_count * sizeof(float), cudaMemcpyHostToDevice);
	delete[] Array;
}

int main()
{

	// User-defined variables
	int Iter, Func, nStocks;
	float T, K, r;
	payoff_t Pricing;

	// Briefly describe the project.
	std::cout << "European-type Rainbow Option pricer" << "\n";
	 
	// User input + validation (through ValidateInput templated function)	
	ValidateInput<int>(Iter, "Number of iterations:         ", [](int X) -> bool { return (X > 0); });
	ValidateInput<int>(nStocks, "Number of stocks:             ", [](int X) -> bool { return (X > 0); });
	ValidateInput<float>(r, "Risk-free rate (% p.a.):      ");
	ValidateInput<float>(T, "Option expiry in years:       ", [](float X) -> bool { return (X > 0.0); });
	ValidateInput<float>(K, "Option strike:                ", [](float X) -> bool { return (X >= 0.0); });
	ValidateInput<int>(Func, PayoffSelect, [](int X) -> bool { return (X >= 1 && X <= PayoffCount); });

	// User input treatment
	r /= 100.f;														// Convert from percentage
	Pricing = Payoffs[Func - 1];									// Select the appropriate Payoff

	// Historical data:
	auto CorrelationMatrix = cudaAlloc(nStocks * nStocks);			// Correlation matrix so that rho(i,i) = 1 and rho(i, j) = Cov(Si, Sj)/sqrt(Var(Si) * Var(Sj))
	auto Sigma = cudaAlloc(nStocks);								// Stocks sigma (volatility p.a.)
	auto S0 = cudaAlloc(nStocks);									// Stocks initial values
	auto q = cudaAlloc(nStocks);									// Stocks Dividend Yields (p.a.)

	parseFile(CorrelationMatrix, CORRELATION_FILE, nStocks * nStocks, ';', "FR-fr");
	parseFile(Sigma, SIGMA_FILE, nStocks, ';', "FR-fr");
	parseFile(S0, S0_FILE, nStocks, ';', "FR-fr");
	parseFile(q, YIELDS_FILE, nStocks, ';', "FR-fr");

	// Simulation data
	auto RandomCorrelated = cudaAlloc(Iter * nStocks);				// Preallocate all correlated random variables into a matrix.
	auto RandomCorrelated_Antithetic = cudaAlloc(Iter * nStocks);	// In order to reduce variance, speeding up the convergence.
	auto MuIto = cudaAlloc(nStocks);
	auto P = cudaAlloc(2 * Iter);
	auto Discount = expf(-r * T);									// European options can only be exercised at expiry

	// Compute the Cholesky decomposition of the Correlation Matrix (ie. M so that MMt = MtM = Corr)
	CholeskyFactor(CorrelationMatrix, nStocks);

	// Compute MuIto[i] = S0[i] * exp((r - q[i] - Sigma[i] * Sigma[i] / 2) * T) 
	InitializeMu(MuIto, r, q, Sigma, S0, nStocks, T);

	// Fill with random normal variables of mean 0. and sd sqrtf(T), then apply correlation.
	RandomFill(RandomCorrelated_Antithetic, Iter * nStocks, T);
	ApplyCorrelation(RandomCorrelated_Antithetic, RandomCorrelated, CorrelationMatrix, Iter, nStocks);

	// Antithetic = (-1) * Correlated Wiener Process
	cudaMemcpy(RandomCorrelated_Antithetic, RandomCorrelated, nStocks * Iter * sizeof(float), cudaMemcpyDeviceToDevice);
	cublasSscal(nStocks * Iter, -1.0f, RandomCorrelated_Antithetic, 1);

	// Generate Paths for both processes
	GeneratePath(RandomCorrelated, MuIto, Sigma, nStocks, Iter);
	GeneratePath(RandomCorrelated_Antithetic, MuIto, Sigma, nStocks, Iter);

	// Price the product accoring to generated paths.
	Pricing(P, RandomCorrelated, K, Discount, nStocks, Iter);
	Pricing(&P[Iter], RandomCorrelated_Antithetic, K, Discount, nStocks, Iter);
	
	// Get average value for the option Price
	auto OptionPrice = cublasSasum(2 * Iter, P, 1) / (2. * Iter);

	// Output it 
	std::cout << "\n" << nStocks << "-Color Rainbow Option Price: " << OptionPrice << "\n";

	// Free memory.
	cudaFree(MuIto);
	cudaFree(CorrelationMatrix);
	cudaFree(RandomCorrelated);
	cudaFree(RandomCorrelated_Antithetic);
	cudaFree(Sigma);
	cudaFree(S0);
	cudaFree(q);
	cudaFree(P);

	// Wait for user input to exit
	(void)getchar();
	return 0;
}