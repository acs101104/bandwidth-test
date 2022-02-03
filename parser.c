#include "parser.h"

#define DEFAULT_DATASIZE 100    // 100 GB

static int str_skip_char(const char *str, const char chr)
{
        int start = 0;

        while (str[start] == chr)
                ++start;

        if (start > strlen(str) - 1)
                return 0;

        return start;
}

static int find_arg_str(const int argc, const char **argv, const char *str)
{
        int found = 0, start;

        for (int i = 1; i < argc; i++) {
                start = str_skip_char(argv[i], '-');
                if (!strcasecmp(&argv[i][start], str))
                        found = 1;
        }

        return found;
}

static int get_arg_str(int argc, char **argv, const char *ref, char **str)
{
        int found = 0, start;
        int len = strlen(ref);
        char *p;

        for (int i = 1; i < argc; i++) {
                start = str_skip_char(argv[i], '-');
                p = &argv[i][start];

                if (!strcmp(p, ref) && argv[i + 1] != NULL && argv[i + 1][0] != '-') {
                        *str = argv[i + 1];
                        found = 1;
                }
                else if (!strncmp(p, ref, len) && p[len] == '=') {
                        *str = &p[len + 1];
                        found = 1;
                }
        }

        return found;
}

static int get_arg_int(int argc, char **argv, const char *ref, int *value)
{
        int found = 0, start;
        int len = strlen(ref);
        char *p;

        for (int i = 1; i < argc; i++) {
                start = str_skip_char(argv[i], '-');
                p = &argv[i][start];

                if (!strcmp(p, ref) && argv[i + 1] != NULL && argv[i + 1][0] != '-') {
                        *value = atoi(argv[i + 1]);
                        found = 1;
                }
                else if (!strncmp(p, ref, len) && p[len] == '=') {
                        *value = atoi(&p[len + 1]);
                        found = 1;
                }
        }

        return found;
}

int get_int_range(char *str, int *buf, int buf_size)
{
        int dash = 0;
        int comma = -1;
        int start, end;

        for (int i = 0, len = 0; i <= strlen(str) && len < buf_size; i++)
                if (!isdigit(str[i]) && str[i] != ',' && str[i] != '-' && str[i]
                                != '\0')
                        return 0;
                else
                        switch (str[i]) {
                        case ',':
                                if (dash) {
                                        start = atoi(&str[comma + 1]);
                                        end = atoi(&str[dash + 1]);
                                        for (int id = start; id <= end; id++, len++)
                                                *buf++ = id;

                                        dash = 0;
                                } else {
                                        *buf++ = atoi(&str[comma + 1]);
                                        ++len;
                                }
                                comma = i;
                                break;
                        case '-':
                                dash = i;
                                break;
                        case '\0':
                                if (dash) {
                                        start = atoi(&str[comma + 1]);
                                        end = atoi(&str[dash + 1]);
                                        for (int i = start; i <= end; i++, len++)
                                                *buf++ = i;
                                } else {
                                        *buf++ = atoi(&str[comma + 1]);
                                        ++len;
                                }
                                return len;
                        }
        return -1;
}

int get_str_range(char *str, char **buf, int buf_size)
{
        int find = 0;
        const char *s = ",";

        char *token = strtok(str, s);

        while (token != NULL && find <= buf_size) {
                *buf++ = token;
                find++;
                token = strtok(NULL, s);
        }

        return find;
}

int parse(int argc, char **argv, Parser *p)
{
    if (find_arg_str(argc, (const char **) argv, "help") || find_arg_str(argc, (const char **) argv, "h"))
            p->help = 1;
    else
            p->help = 0;

    if (!get_arg_int(argc, argv, "size", &p->size) && !get_arg_int(argc, argv, "s", &p->size))
            p->size = DEFAULT_DATASIZE;

    if (get_arg_str(argc, argv, "id", &p->id))
            p->name = NULL;
    else if (get_arg_str(argc, argv, "name", &p->name))
            p->id = NULL;
    else
            p->id = p->name = NULL;

    if (!get_arg_int(argc, argv, "src", &p->src))
            p->src = 0;

    if (!get_arg_int(argc, argv, "dst", &p->dst))
            p->dst = 1;

    if (find_arg_str(argc, (const char **) argv, "h2d"))
            p->method = h2d;
    else if (find_arg_str(argc, (const char **) argv, "d2h"))
            p->method = d2h;
    else if (find_arg_str(argc, (const char **) argv, "bid"))
            p->method = hbd;
    else if (find_arg_str(argc, (const char **) argv, "p2p"))
            p->method = p2p;
    else if (find_arg_str(argc, (const char **) argv, "bidp2p"))
            p->method = pbp;
    else
            p->method = unknown;

    return 0;
}
