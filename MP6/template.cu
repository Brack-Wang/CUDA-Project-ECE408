// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here
// Kernel Code:
// Convert the image from RGB to GrayScale
__global__
void RGBToGrayScale(unsigned char * grayImage, unsigned char * rgbImage, int width, int height) {
  int Col = threadIdx.x + blockIdx.x * blockDim.x;
  int Row = threadIdx.y + blockIdx.y * blockDim.y;
  int Channels = 3;
  if (Col < width && Row < height) {
    int grayOffset = Row * width + Col;
    int rgbOffset = grayOffset * Channels;
    unsigned char r = rgbImage[rgbOffset];
    unsigned char g = rgbImage[rgbOffset + 1];
    unsigned char b = rgbImage[rgbOffset + 2];
    grayImage[grayOffset] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

// Compute the histogram of grayImage
__global__
void histo_kernel(unsigned char * buffer, int size, float *histo) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while  (i < size) {
    atomicAdd(&(histo[buffer[i]]), 1);
    i += stride;
  }
}

// Compute the Cumulative Distribution Function of histogram
__global__ 
void scan(float *input, float *output, int len, int size) {
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start  =  2 * blockDim.x * blockIdx.x;
  if ( start + t < len) {
    partialSum[t]  = 1.0 * input[start + t] / size;
  }
  else {
    partialSum[t] = 0.0f;
  }
  if ( start + t + blockDim.x < len) {
    partialSum[t + blockDim.x] = 1.0 * input[start + t + blockDim.x] / size;
  }
  else {
    partialSum[t + blockDim.x] = 0.0f;
  }
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
      partialSum[index] += partialSum[index - stride];
    }
    stride = stride * 2;
  }
  int stride2 = BLOCK_SIZE / 2;
  while (stride2 > 0) {
    __syncthreads();
    int index2 = (t + 1) * stride2 * 2 - 1;
    if ((index2 + stride2) < 2 * BLOCK_SIZE) {
      partialSum[index2 + stride2] += partialSum[index2];
    }
    stride2 = stride2/2;
  }
  __syncthreads();
  output[start + t] = partialSum[t];
  output[start + t + blockDim.x] = partialSum[t + blockDim.x];
}

// Define the histogram equalization function
float clamp(float x, float start, float end) {
  return min(max(x, start), end);
}

float correct_color(unsigned char hostUcharImage, float *Cdf) {
  int val = (int)hostUcharImage;
  int cdfmin = (int)100000;
  for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
    if (Cdf[i] < cdfmin) {
      cdfmin = Cdf[i];
    }
  }
  return clamp(255 * (Cdf[val] - cdfmin) / (1.0 - cdfmin), 0, 255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  // Param Definition: 
  // Convert the image from RGB to GrayScale
  unsigned char *deviceUcharImage;
  unsigned char *deviceGrayImage;
  // Compute the histogram of grayImage
  float *deviceHistogram;
  // Compute the Cumulative Distribution Function of histogram
  float *deviceCdf;
  float *hostCdf;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // Host Code: 
  int rgbLength = int(imageWidth * imageHeight * imageChannels);
  int grayLength = int(imageWidth * imageHeight);
  int rgbSize = rgbLength * sizeof(unsigned char);
  int graySize = grayLength * sizeof(unsigned char);

  // Cast the image from float to unsigned char
  unsigned char hostUcharImage[rgbLength];
  for (int i = 0; i < rgbLength; i++) {
    hostUcharImage[i] = (unsigned char)255 * hostInputImageData[i];
  }

  // Convert the image from RGB to GrayScale
  cudaMalloc((void **) &deviceUcharImage, rgbSize);
  cudaMalloc((void **) &deviceGrayImage, graySize);
  cudaMemcpy(deviceUcharImage, hostUcharImage, rgbSize, cudaMemcpyHostToDevice);
  dim3 gridDimRGBToGray(ceil(1.0 * imageWidth / BLOCK_SIZE), ceil(1.0 * imageHeight / BLOCK_SIZE), 1);
  dim3 blockDimRGBToGray(BLOCK_SIZE, BLOCK_SIZE, 1);
  RGBToGrayScale<<<gridDimRGBToGray, blockDimRGBToGray>>>(deviceGrayImage, deviceUcharImage, imageWidth, imageHeight);

  // Compute the histogram of grayImage
  float hostHistogram[HISTOGRAM_LENGTH] = {0};
  int histoSize = HISTOGRAM_LENGTH * sizeof(float);
  cudaMalloc((void **) &deviceHistogram, histoSize);
  cudaMemcpy(deviceHistogram, hostHistogram, histoSize, cudaMemcpyHostToDevice);
  int gridDimHisto = ceil(1.0 * grayLength / BLOCK_SIZE);
  int blockDimHisto = BLOCK_SIZE;
  histo_kernel<<<gridDimHisto, blockDimHisto>>>(deviceGrayImage, grayLength, deviceHistogram);

  // Compute the Cumulative Distribution Function of histogram
  hostCdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  int cdfSize = HISTOGRAM_LENGTH * sizeof(float);
  cudaMalloc((void **) &deviceCdf, cdfSize);
  int gridDimCdf = ceil(1.0 * HISTOGRAM_LENGTH / BLOCK_SIZE);
  int blockDimCdf = BLOCK_SIZE;
  scan<<<gridDimCdf, blockDimCdf>>>(deviceHistogram, deviceCdf, HISTOGRAM_LENGTH, grayLength);
  cudaMemcpy(hostCdf, deviceCdf, cdfSize, cudaMemcpyDeviceToHost);
  int block_tmp;
  for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
    block_tmp = i / (2.0 * BLOCK_SIZE);
    if (block_tmp > 0) {
      hostCdf[i] += hostCdf[block_tmp  * 2 * BLOCK_SIZE - 1];
    }
  }

  // Apply the histogram equalization function
  for (int i = 0; i < rgbLength; i++) {
    hostUcharImage[i] = correct_color(hostUcharImage[i], hostCdf);
  }

  // Cast back to float
  for (int i = 0; i < rgbLength; i++) {
    hostOutputImageData[i] = (float)(hostUcharImage[i]/255.0);
  }

  // outputImage = hostOutputImageData;
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceGrayImage);
  cudaFree(deviceUcharImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
