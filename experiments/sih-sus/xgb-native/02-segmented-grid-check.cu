// Standalone CUDA check of segmented block mapping and fixed-point accumulation.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
constexpr unsigned kThreads = 256;
constexpr unsigned kNodes = 32;
constexpr unsigned kColumns = 24;
constexpr unsigned kBinsPerColumn = 8;
constexpr unsigned kBins = kColumns * kBinsPerColumn;
constexpr unsigned kRows = 3107;
constexpr unsigned kTotalNodes = 71;

struct Pair {
  unsigned long long grad;
  unsigned long long hess;
};
struct Node {
  unsigned begin;
  unsigned count;
  unsigned id;
};
struct Launch {
  Node nodes[kNodes];
  unsigned offsets[kNodes + 1];
  unsigned size{};
};

void Check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << cudaGetErrorString(result) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

__global__ void BatchHistogram(const __grid_constant__ Launch batch, const Pair* gradients,
                               const std::uint8_t* bins, Pair* histograms) {
  unsigned low = 0, high = batch.size;
  while (high - low > 1) {
    unsigned mid = (low + high) / 2;
    if (batch.offsets[mid] <= blockIdx.x) {
      low = mid;
    } else {
      high = mid;
    }
  }
  auto node = batch.nodes[low];
  unsigned blocks = batch.offsets[low + 1] - batch.offsets[low];
  unsigned begin = (blockIdx.x - batch.offsets[low]) * blockDim.x + threadIdx.x;
  for (unsigned i = begin; i < node.count * kColumns; i += blockDim.x * blocks) {
    unsigned row = node.begin + i / kColumns;
    unsigned col = i % kColumns;
    std::uint8_t bin = bins[row * kColumns + col];
    if (bin == kBinsPerColumn) {
      continue;
    }
    Pair& out = histograms[node.id * kBins + col * kBinsPerColumn + bin];
    atomicAdd(&out.grad, gradients[row].grad);
    atomicAdd(&out.hess, gradients[row].hess);
  }
}
}  // namespace

int main() {
  Pair* gradients{};
  std::uint8_t* bins{};
  Pair* actual{};
  Check(cudaMallocManaged(&gradients, sizeof(Pair) * kRows));
  Check(cudaMallocManaged(&bins, sizeof(std::uint8_t) * kRows * kColumns));
  Check(cudaMallocManaged(&actual, sizeof(Pair) * kBins * kTotalNodes));
  std::fill_n(actual, kBins * kTotalNodes, Pair{});
  std::vector<Pair> expected(kBins * kTotalNodes);
  for (unsigned row = 0; row < kRows; ++row) {
    gradients[row] = {
        static_cast<unsigned long long>(static_cast<int>(row % 31) - 14), row % 23 + 1};
    for (unsigned col = 0; col < kColumns; ++col) {
      bins[row * kColumns + col] = (row + col * 17) % (kBinsPerColumn + 1);
    }
  }

  Launch batch{};
  unsigned launches = 0, nonempty = 0;
  auto flush = [&] {
    if (batch.size != 0) {
      BatchHistogram<<<batch.offsets[batch.size], kThreads>>>(batch, gradients, bins, actual);
      Check(cudaGetLastError());
      Check(cudaDeviceSynchronize());
      batch.size = 0;
      ++launches;
    }
  };
  for (unsigned id = 0; id < kTotalNodes; ++id) {
    unsigned count = id % 8 == 0 ? 0 : 1 + (id * 301) % 101;
    if (count == 0) {
      continue;
    }
    Node node{(id * 113) % (kRows - count), count, id};
    for (unsigned row = node.begin; row < node.begin + count; ++row) {
      for (unsigned col = 0; col < kColumns; ++col) {
        auto bin = bins[row * kColumns + col];
        if (bin != kBinsPerColumn) {
          auto& out = expected[id * kBins + col * kBinsPerColumn + bin];
          out.grad += gradients[row].grad;
          out.hess += gradients[row].hess;
        }
      }
    }
    unsigned blocks = std::min(3U, (count * kColumns + kThreads - 1) / kThreads);
    batch.nodes[batch.size] = node;
    batch.offsets[batch.size + 1] = batch.offsets[batch.size] + blocks;
    if (++batch.size == kNodes) {
      flush();
    }
    ++nonempty;
  }
  flush();
  Check(cudaDeviceSynchronize());
  for (unsigned i = 0; i < expected.size(); ++i) {
    if (actual[i].grad != expected[i].grad || actual[i].hess != expected[i].hess) {
      std::cerr << "Mismatch at histogram entry " << i << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << expected.size() << " exact pairs across " << kTotalNodes << " nodes ("
            << nonempty << " nonempty), " << launches << " kernel launches\n";
  Check(cudaFree(actual));
  Check(cudaFree(bins));
  Check(cudaFree(gradients));
}
