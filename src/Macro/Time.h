#define __now__ std::chrono::high_resolution_clock::now()
#define __duration__(start, end) std::chrono::duration<double>(end - start).count()
