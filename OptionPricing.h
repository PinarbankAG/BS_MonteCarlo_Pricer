#pragma comment(lib,"curand.lib")
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cusolver.lib")

#include <device_launch_parameters.h>
#include <random>
#include <iostream> 
#include <string>
#include <sstream>
#include <curand.h>
#include <cublas.h>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <locale>

// Trick for using cuSOLVER in Visual Studio...

extern "C" {
	struct cusolverDnContext;
	typedef struct cusolverDnContext *cusolverDnHandle_t;
	typedef enum {
		CUSOLVER_STATUS_SUCCESS = 0,
		CUSOLVER_STATUS_NOT_INITIALIZED = 1,
		CUSOLVER_STATUS_ALLOC_FAILED = 2,
		CUSOLVER_STATUS_INVALID_VALUE = 3,
		CUSOLVER_STATUS_ARCH_MISMATCH = 4,
		CUSOLVER_STATUS_MAPPING_ERROR = 5,
		CUSOLVER_STATUS_EXECUTION_FAILED = 6,
		CUSOLVER_STATUS_INTERNAL_ERROR = 7,
		CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
		CUSOLVER_STATUS_NOT_SUPPORTED = 9,
		CUSOLVER_STATUS_ZERO_PIVOT = 10,
		CUSOLVER_STATUS_INVALID_LICENSE = 11
	} cusolverStatus_t;

	cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
	cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
	cusolverStatus_t cusolverDnSpotrf(
		cusolverDnHandle_t handle,
		cublasFillMode_t uplo,
		int n,
		float *A,
		int lda,
		float *Workspace,
		int Lwork,
		int *devInfo);
	cusolverStatus_t cusolverDnSpotrf_bufferSize(
		cusolverDnHandle_t handle,
		cublasFillMode_t uplo,
		int n,
		float *A,
		int lda,
		int *Lwork);
}

// Threads per Block

#define TPB 256

// Parameters file definition 

#define CORRELATION_FILE "CORREL.csv"
#define SIGMA_FILE "SIGMA.csv"
#define S0_FILE "S0.csv"
#define YIELDS_FILE "YIELDS.csv"

// Kernel functions

__global__ void InitializeMuKernel(float *, const float, const float *, const float *, const float *, const int, const float);

__global__ void GeneratePathKernel(float * __restrict__, const float * __restrict__, const float * __restrict__, const int, const int);

__global__ void OnMax(float * __restrict__, const float * __restrict__, const int, const int);

__global__ void OnMin(float * __restrict__, const float * __restrict__, const int, const int);

__global__ void Call(float *, const float, const float, const int);

__global__ void Put(float *, const float, const float, const int);

__global__ void Best(float *, const float, const float, const int);

__global__ void MakeLower(float *, int);

// Utility functions

inline float * cudaAlloc(size_t);

void CholeskyFactor(float *, const int);

inline void ApplyCorrelation(const float * __restrict__, float * __restrict__, float * __restrict__, const int, const int);

void RandomFill(float *, const int, const float);

inline void GeneratePath(float * __restrict__, const float * __restrict__, const float * __restrict__, const int, const int);

inline void InitializeMu(float *, const float, const float *, const float *, const float *, const int, const float);

// Payoffs Implementation

typedef void(*payoff_t)(float *, const float *, const float, const float, const int, const int);

inline void CallOnMax(float *, const float *, const float, const float, const int, const int);

inline void CallOnMin(float *, const float *, const float, const float, const int, const int);

inline void PutOnMax(float *, const float *, const float, const float, const int, const int);

inline void PutOnMin(float *, const float *, const float, const float, const int, const int);

inline void BestOfAssetsOrCash(float *, const float *, const float, const float, const int, const int);

const payoff_t Payoffs[] = { CallOnMax, CallOnMin, PutOnMax, PutOnMin, BestOfAssetsOrCash };

const std::string PayoffSelect = "Option Payoff:\n"
"       1. Call On Max\n"
"       2. Call On Min\n"
"       3. Put On Max\n"
"       4. Put On Min\n"
"       5. Best Asset Or Cash: ";

const int PayoffCount = 5;

// Parser Implementation

inline bool loadFile(const char *, std::string&, const char);

void fillFloats(float *, const int, std::string&, const char *);

void parseFile(float *, const char *, const int, const char, const char *);

// Input Validation

template<class T>
void ValidateInput(T& out, std::string const& Msg) {
	std::string extractedLine;
	bool validInput;
	do {
		std::cout << "\n";
		std::cout << Msg;
		std::getline(std::cin, extractedLine);
		std::istringstream iss(extractedLine);
		iss >> out >> std::ws;
		validInput = !iss.fail() && iss.eof();
	} while (!validInput);
}

template<class T, class Pred>
void ValidateInput(T& out, std::string const& Msg, Pred predicate) {
	do {
		ValidateInput(out, Msg);
	} while (!predicate(out));
}