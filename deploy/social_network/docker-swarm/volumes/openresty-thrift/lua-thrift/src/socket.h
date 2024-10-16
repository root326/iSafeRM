
#ifndef LUA_THRIFT_SOCKET_H
#define LUA_THRIFT_SOCKET_H

#include <sys/socket.h>

#ifdef _WIN32
#else
typedef int t_socket;
typedef t_socket* p_socket;
#endif

enum {
  SUCCESS = 0,
  TIMEOUT = -1,
  CLOSED = -2,
};
typedef int T_ERRCODE;

static const char * TIMEOUT_MSG = "Timeout";
static const char * CLOSED_MSG = "Connection Closed";

typedef struct sockaddr t_sa;
typedef t_sa * p_sa;

T_ERRCODE socket_create(p_socket sock, int domain, int type, int protocol);
T_ERRCODE socket_destroy(p_socket sock);
T_ERRCODE socket_bind(p_socket sock, p_sa addr, int addr_len);
T_ERRCODE socket_get_info(p_socket sock, short *port, char *buf, size_t len);
T_ERRCODE socket_send(p_socket sock, const char *data, size_t len, int timeout);
T_ERRCODE socket_recv(p_socket sock, char *data, size_t len, int timeout,
                      int *received);

T_ERRCODE socket_setblocking(p_socket sock);
T_ERRCODE socket_setnonblocking(p_socket sock);

T_ERRCODE socket_accept(p_socket sock, p_socket sibling,
                        p_sa addr, socklen_t *addr_len, int timeout);
T_ERRCODE socket_listen(p_socket sock, int backlog);

T_ERRCODE socket_connect(p_socket sock, p_sa addr, int addr_len, int timeout);

const char * tcp_create(p_socket sock);
const char * tcp_destroy(p_socket sock);
const char * tcp_bind(p_socket sock, const char *host, unsigned short port);
const char * tcp_send(p_socket sock, const char *data, size_t w_len,
                      int timeout);
const char * tcp_receive(p_socket sock, char *data, size_t r_len, int timeout);
const char * tcp_raw_receive(p_socket sock, char * data, size_t r_len,
                             int timeout, int *received);

const char * tcp_listen(p_socket sock, int backlog);
const char * tcp_accept(p_socket sock, p_socket client, int timeout);

const char * tcp_connect(p_socket sock, const char *host, unsigned short port,
                         int timeout);

#endif
