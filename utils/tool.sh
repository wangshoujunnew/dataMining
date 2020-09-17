# 批量杀掉任务
function mykill(){
    set -x
    ps aux | grep $1 | awk '{print $2}' | xargs -I{} kill -9 {}
    set +x
}

# 设置所有的网络走此代理
export ALL_PROXY=socks5://100.80.129.65:1080
unset ALL_PROXY
export TERM=linux


