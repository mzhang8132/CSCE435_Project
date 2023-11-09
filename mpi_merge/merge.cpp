#include <algorithm>


/*  Work in progress
a list of sources:
- (main one) https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html
- https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c
- https://stackoverflow.com/questions/61910607/parallel-merge-sort-using-mpi
*/

float* merge(float*, float*, float*, size_t);

float* mergeSort(int height, int id, float localArray[], size_t size, MPI_Comm comm, float globalArray[])
{
  int parent, rightChild, myHeight;
  float *half1, *half2, *mergeResult;

  myHeight = 0;
  std::sort(&localArray[0], &localArray[size - 1]); // sort small, local array using sequential means
  half1 = localArray;

  while (myHeight < height) {  
    parent = (id & (~(1 << myHeight)));

    if (parent == id) {
      rightChild = (id | (1 << myHeight));


      half2 = new float[size];
      MPI_Recv(half2, size, MPI_INT, rightChild, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


      mergeResult = new float[size];

      mergeResult = merge(half1, half2, mergeResult, size);

      half1 = mergeResult;
      size = size * 2;

      delete[] half2;
      mergeResult = NULL;

      myHeight++;

    } else {
        
      MPI_Send(half1, size, MPI_INT, parent, 0, MPI_COMM_WORLD);
      if (myHeight != 0) delete[] half1;
      myHeight = height;
    }
  }

  if (id == 0) globalArray = half1;
  return globalArray;
}

int main(int argc, char** argv)
{
  int numProcs, id, globalArraySize, localArraySize, height;
  float *localArray, *globalArray;
  double startTime, localTime, totalTime;
  double zeroStartTime, zeroTotalTime, processStartTime, processTotalTime;
  
  int length = -1;
  char myHostName[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  MPI_Get_processor_name(myHostName, &length);

  if (id == 0) {
    globalArray = new float[globalArraySize];
  }

  localArraySize = globalArraySize / numProcs;
  localArray = new float[localArraySize];
  MPI_Scatter(globalArray, localArraySize, MPI_INT, localArray, localArraySize, MPI_INT, 0, MPI_COMM_WORLD);

  startTime = MPI_Wtime();
  if (id == 0) {
    zeroStartTime = MPI_Wtime();
    globalArray = mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, globalArray);
    zeroTotalTime = MPI_Wtime() - zeroStartTime;
    printf("Process #%d of %d on %s took %f seconds \n", id, numProcs, myHostName, zeroTotalTime);
  } else {
    processStartTime = MPI_Wtime();
    mergeSort(height, id, localArray, localArraySize, MPI_COMM_WORLD, NULL);
    processTotalTime = MPI_Wtime() - processStartTime;
    printf("Process #%d of %d on %s took %f seconds \n", id, numProcs, myHostName, processTotalTime);
  }
  localTime = MPI_Wtime() - startTime;
  MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (id == 0) {
    printf("Sorting %d integers took %f seconds \n", globalArraySize, totalTime);
    delete[] globalArray;
  }

  delete[] localArray;
  MPI_Finalize();
  return 0;
}