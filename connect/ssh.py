# https://pypi.org/project/sshtunnel/ 下载 ssh 隧道
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import paramiko

# 无法直接连接的原因 Authentication refused: bad ownership or modes for directory /home/shoujunw
# https://blog.csdn.net/lovebyz/article/details/87916317 查看sshd中是否其中授权, 改完目录权限成功登陆

# 连接方法 ssh -p 端口 用户@ip地址
# 将跳板机的公钥推送到远程服务器上 ssh-copy-id -i ~/.ssh/id_rsa.pub 192.168.10.211
# wanted were added 添加成功标识
private_key = paramiko.RSAKey.from_private_key_file('c:/Users/shoujunw/.ssh/windows')
server = SSHTunnelForwarder(
    local_bind_address=('0.0.0.0', 22),  # 本地端口启动10022服务, 跳转到跳板机的22服务,然后连接108的22端口

    ssh_address_or_host=('', 22),  # 跳转机
    ssh_pkey="c:/Users/shoujunw/.ssh/windows",
    # ssh_username="shoujunw",
    # ssh_password="fGLau6Lgmb7BhpSy",
    # ssh_pkey="d:/key/108",
    # ssh_username="",
    # ssh_pkey="d:/key/24jump",
    # ssh_private_key_password="secret",

    # ssh_private_key_password="fGLau6Lgmb7BhpSy",
    remote_bind_address=('', 22),  # 访问的远程主机
)

server.start()
#
print(server.local_bind_port)  # show assigned local port
# work with `SECRET SERVICE` through `server.local_bind_port`.

# server.stop()

# 监听远程服务器端口443, 且443处于打开状态
# with sshtunnel.open_tunnel(
#     (REMOTE_SERVER_IP, 443),
#     ssh_username="",
#     ssh_pkey="/var/ssh/rsa_key",
#     ssh_private_key_password="secret",
#     remote_bind_address=(PRIVATE_SERVER_IP, 22),
#     local_bind_address=('0.0.0.0', 10022)
# ) as tunnel:
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect('127.0.0.1', 10022)
#     # do some operations with client session
#     client.close()
#
# print('FINISH!')

# 108notebook
# with sshtunnel.open_tunnel(
#         ("", 10010),
#         # ssh_pkey="/var/ssh/rsa_key",
#         # ssh_private_key_password="secret",
#         ssh_address_or_host=('172.31.90.24', 22),  # 跳转机
#         ssh_username="shoujunw",
#         ssh_password="fGLau6Lgmb7BhpSy",
#         remote_bind_address=("172.31.84.108", 22),
#         local_bind_address=('0.0.0.0', 10022)
# ) as tunnel:
#     client = paramiko.SSHClient()
#     # client.load_system_host_keys()
#     # client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect('127.0.0.1', 10022)
#     # # do some operations with client session
#     # client.close()
#
# print('FINISH!')
