# CSCE 435 Group project

## 1. Group members:
1. Evan Burriola
2. Min Zhang
3. Cole McAnelly
4. Saddy Khakimova

All communication between team members will be coordinated through Discord.

---

## 2. _due 10/25_ Project topic

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For this project we will be comparing both merge sort and quick sort algorithms. Each sorting algorithm will be implemented using both CUDA and MPI across different number of threads in order to compare their parallel performances against their respective sequential versions. 

### MPI Merge Sort

procedure MERGESORT(A, left, right)
begin
    if left < right then
    begin
        middle = (left + right) / 2

        MERGESORT(A, left, middle)
        MERGESORT(A, middle + 1, right)
        MERGE(A, left, middle, right)
    end if
end MERGESORT

Steps to be done in main:
- read in array
- initialize MPI
- split array into chunks for each thread
- send array chunk to each corresponding thread
- do merge sort on each thread
- gather all sorted chunks from each thread
- have one thread perform one last merge sort

### CUDA Merge Sort

procedure MERGESORT(A)
begin
    allocate memory
    Memcpy host to device
    MERGESORT_STEP<<<blocks, threads>>>(A, j, k)
    Memcpy device to host
    free allocated memory
end MERGESORT

### MPI Quick Sort

procedure QUICKSORT(A, start, end)
begin
    if start < end then
    begin
        pivot = A[start]
        index = start
        for i = start+1 to end do
            if A[i] < pivot then
            begin
                index = index + 1
                swap(A[index], A[i])
            end if
        swap(A[start], A[index])
        QUICKSORT(A, start, index)
        QUICKSORT(A, index + 1, end)
    end if
end QUICKSORT

Steps to be done in main:
- read in array
- initialize MPI
- split array into chunks for each thread
- send array chunk to each corresponding thread
- do quicksort on each thread
- pair up the resulting sorted chunks and continute to run quicksort until the entire array is sorted and recombined

### CUDA Quick Sort

procedure QUICKSORT(A)
begin
    allocate memory
    Memcpy host to device
    QUICKSORT_STEP<<<blocks, threads>>>(A, j, k)
    Memcpy device to host
    free allocated memory
end QUICKSORT