#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

typedef enum { h2d, d2h, hbd, p2p, pbp, unknown } transfer;

typedef struct {
    char *id;
    char *name;
    int size;
    int help;
    int src;
    int dst;
    transfer method;
} Parser;

#ifdef __cplusplus
extern "C" {
#endif

int parse(int argc, char **argv, Parser *);
int get_int_range(char *str, int *ranges, int size);
int get_str_range(char *str, char **buf, int buf_size);

#ifdef __cplusplus
}
#endif

#endif
