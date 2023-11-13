#include <iostream>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

// HELPER FUNCTIONS

int random(int) { return (int)rand() % (int)10; }

float random(float) { return (float)rand() / (float)RAND_MAX; }

template<typename T>
inline void print_array(T* arr, size_t length, const char* title = "", int id = -1)
{
  std::cout << title;
  if (id != -1) std::cout << "\tID:" << id;
  std::cout << "\n{";
  for (int i = 0; i < length; ++i) std::cout << " " << arr[i] << ",";
  std::cout << "\b}\n";
}

template<typename T>
inline void fill_array(T* arr, size_t len, int option)
{ 
  std::cout << "Filling array with option:" << option << "\n";
  CALI_MARK_BEGIN("data_init");
  srand(0);
  switch (option)
  {
    case 1: // Random
      for (int i = 0; i < len; ++i) arr[i] = random(T());    break;
    case 2: // Sorted
      for (int i = 0; i < len; ++i) arr[i] = T(i);           break;
    case 3: // Reverse Sorted
      for (int i = 0; i < len; ++i) arr[i] = T(len-1-i);  break;
    case 4: // 1% Perturbed
      for (int i = 0; i < len; ++i) arr[i] = T(i);
      for (int i = 0; i < len / 100; ++i) arr[rand() % len] = random(T());
      break;
    default: break;
  }
  CALI_MARK_END("data_init");
}