from odo.backends.tests.test_ssh import paramiko
from sshtunnel import SSHTunnelForwarder
import sys
# 建立端口转发, 本地端口， 选择的跳板机， 远程机器
def build_port_trans(local_port, jump, remote_ip_and_port):
    jump_server = {"24": {
                        "ip": '172.31.90.24',
                        "password": "fGLau6Lgmb7BhpSy"
                    },
                    "46": {
                        "ip": "172.31.90.46",
                        "password": "INYAWq!ZWvpOzW-"
                    }

    # INYAWq\!ZWvpOzW\-
    }.get(jump)
    print("=====")
    print(jump_server)
    local_bind_addresses = [('0.0.0.0', port) for port in local_port]
    print(local_bind_addresses)
    print(remote_ip_and_port)
    # private_key = paramiko.RSAKey.from_private_key_file("/Users/tjuser/Downloads/id_rsa")
    server = SSHTunnelForwarder(
        local_bind_addresses=local_bind_addresses,
        ssh_address_or_host=(jump_server["ip"], 22),  # 跳转机
        ssh_username="shoujunw",
        ssh_password="fGLau6Lgmb7BhpSy",
        remote_bind_addresses=remote_ip_and_port,  # 访问的远程主机
    )

    server.start()
    print(server.local_bind_port)

jump_enum = "24"
build_port_trans([22108], jump_enum, [("172.31.84.108", 22)])