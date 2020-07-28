import yaml as yaml
import sys

from sshtunnel import SSHTunnelForwarder

jump = sys.argv[1] if len(sys.argv) > 1 else "jump46"

config = yaml.load(open("ssh.yml"))
config = config[jump]
print(config)

def get_local_bind_addresses(config):
    locals, remotes = [], []
    for remote_ip,vlist in config.items():

        if not '.' in remote_ip:
            continue ## 只处理ip

        for v in vlist:
            if type(v) == int:
                locals.append(('0.0.0.0', v))
                remotes.append((remote_ip, v))
            else:
                vs = v.split(":")
                if vs[-1] != "False":
                    locals.append(('0.0.0.0', int(vs[1])))
                    remotes.append((remote_ip, int(vs[0])))
    print(locals, remotes)
    return locals, remotes

locals, remotes = get_local_bind_addresses(config)

server2 = SSHTunnelForwarder(
        local_bind_addresses=locals,
        ssh_address_or_host=(config["jumpip"], 22),  # 跳转机
        ssh_pkey=config["ssh_pkey"] if "ssh_pkey" in config.keys() else None,
        ssh_username="shoujunw",
        ssh_private_key_password= config["password"] if "password" in config.keys() else None,
        remote_bind_addresses=remotes
    )

server2.start()
