/**
 * buffer.c: a small utility that buffers file reads for wiktextract.
 *
 * As I pointed out in the original project ( see
 * https://github.com/tatuylonen/wiktextract/issues/23 ) the bzip extractor
 * code makes references to a "buffer" program but does not explain where it
 * located nor where to build it or what the source code is.
 * Given the context it is quite easy to tell what it does.
 *
 * This snippet simply provides a placeholder for that.
 * Once this porgram is compiled it is then moved to the PATH.
 *
 * And then wiktextract can be safely used :)
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define MEGA 1000000
int main(int argc, char** argv)
{
    // The program expects a "-m" parameter that provides the buffer size.
    if(!(argc == 3 && strcmp(argv[1], "-m") == 0))
        return 1;

    size_t buf_size;

    sscanf(argv[2], "%ludM", &buf_size);
    buf_size = buf_size * MEGA;
    setvbuf(stdin, NULL, _IONBF, 0);
    
    char* memblock = malloc(buf_size);

    int read_size;
    
    // For performance reasons, we use direct UNIX syscalls
    while((read_size = read(0, memblock, buf_size)))
        write(1, memblock, read_size);

    free(memblock);
}

