# encoding=utf-8
import sys, socket, time, threading

loglock = threading.Lock()


def log(msg):
    loglock.acquire()
    try:
        print('[%s]: \n%s\n' % (time.ctime(), msg.strip()))
        sys.stdout.flush()
    finally:
        loglock.release()


class PipeThread(threading.Thread):
    def __init__(self, source, target):
        threading.Thread.__init__(self)
        self.source = source
        self.target = target

    def run(self):
        while True:
            try:
                data = self.source.recv(1024)
                log(data)
                if not data: break
                self.target.send(data)
            except:
                break
        log('PipeThread done')


class Forwarding(threading.Thread):
    def __init__(self, port, targethost, targetport):
        threading.Thread.__init__(self)
        self.targethost = targethost
        self.targetport = targetport
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.listen(10)  # 守护进程

    def run(self):
        while True:
            try:
                client_fd, client_addr = self.sock.accept()
                target_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                target_fd.connect((self.targethost, self.targetport))
                log('new connect')
                # two direct pipe
                PipeThread(target_fd, client_fd).start()
                PipeThread(client_fd, target_fd).start()
            except:
                print("连接到远程异常")


if __name__ == '__main__':
    print('Starting')
    # 端口转发配置
    config = [
        {"remotePort": 6070, "localPort": 16070, "remoteIp": "172.31.84.109", "desc": "rank 解释工具"},
        # 存在的问题, 对一个大系统网站不友好, 域名总是改变, 可以通过修改 host, 将某个域名直接指向到某台主机上 ok 完美
        # 使用代理的方式, 在本地开通一个代理 50011, 然后 50011 是隐射到 46 跳板机的, 当有流量访问到本地 50011 端口, 则直接使用跳板机访问
        # 浏览器上使用 switch proxy
        {"remotePort": 443, "localPort": 1443, "remoteIp": "10.95.149.43", "desc": "去哪儿 wiki"},
    ]

    for cf in config:
        try:
            port = cf.get("localPort", cf["remotePort"])
            targetport = cf["remotePort"]
            targethost = cf["remoteIp"]
            print((port, targethost, targetport))
            Forwarding(port, targethost, targetport).start()
            print("success")
        except (ValueError, IndexError):
            print('Usage: %s port targethost [targetport]' % sys.argv[0])
            continue
            # sys.exit(1)
        # sys.stdout = open('forwaring.log', 'w')
