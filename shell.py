import subprocess
from unittest import TestCase


def run(cmd):
    subprocess.run(cmd, shell=True)

"""
测试 shell 命令
"""
class ShellTest(TestCase):
    def testAwk(self):
        print("test Awk, 整行不包含 x 且第二列匹配到 2, 然后输出第三列和文本的长度, 最后格式输出一下, 并在开头定义一个 map, 在$3 中用 replace2 替换所有的匹配到的3 字符")
        run(""" echo 1 2 y3x | awk ' !/x/ && $2 ~/2/ 
                    BEGIN{
                        testMap["name"]="shoujunw";testMap["age"]=10
                    } {gsub(".*3.*", "replace2", $3);print $3 , length}
                    END{printf "this value is %.2f, %s", 0.1111, testMap["name"]}' """)
        print("\nawk 便利 map 或者数组的方式, 时间的应用")
        run(""" echo 1 2 3 | awk 'BEGIN{
                testMap["name"]="shoujunw"; testMap["age"]=10}END{for(n in testMap){print testMap[n]}}' """)
        # test="$test" 获取外部变量 变量名与值放到’{action}’后面。

    def testMysql(self):
        print("导出数据库的表结构 ")
        print("mysqldump -uroot -pdbpasswd -d dbname >db.sql;") # 不加 d 的时候表示导出结构+数据
        print("执行某个 sql")


    def testDiff(self):
        print("test diff")
        print("并排输出 -W 指定兰宽")
        run("diff ~/tmp/1.txt ~/tmp/2.txt -y -W 50")
    def testSort(self):
        print("排序,然后输出到原文件中, 忽略空行")
        run("sort ~/tmp/2.txt -o ~/tmp/2.txt")
    def testNet(self):
        print("查看网络流量")
        run("ifstat")
        print("netstat -tunpl | grep 端口号") # 查看端口被那个程序占用
        print("netstat -na")
        """ 查看端口的连接数量
        1)统计80端口连接数
        netstat -nat|grep -i "80"|wc -l
        
        2）统计httpd协议连接数
        ps -ef|grep httpd|wc -l
        
        3）、统计已连接上的，状态为“established
        netstat -na|grep ESTABLISHED|wc -l
        
        4)、查出哪个IP地址连接最多,将其封了.
        netstat -na|grep ESTABLISHED|awk {print $5}|awk -F: {print $1}|sort|uniq -c|sort -r +0n
        
        netstat -na|grep SYN|awk {print $5}|awk -F: {print $1}|sort|uniq -c|sort -r +0n
        """

    def testXargs(self):
        print("test xargs 命令, xargs 默认将多行转为单行, 或者选择没 3 行当做一行输入给下一个命令")
        print("-I 指定一个{} 的替换字符")
        run("ls -l | awk '{print $NF}' | xargs -n100 -I{} cat {}")
        run("echo train.date,v2_feature.txt | xargs -d,") # mac 的系统和 linux 不太一样

    def testFind(self):
        print("test find 命令, 文件以 py 或者 pdf 结尾")
        run(""" find . \( -iname "*.py" -o -iname "*.md" \) """)
        run(""" find . -iregex ".*\(\.txt\|\.pdf\)$" """) # linux 上可行
        print("查找比 文件 1 新, 比文件 2 旧的文件  ! -newer EstimatorWD.py, 大于 1M 的文件, -atime -7 最近 7 天被访问过的文件")
        # 7: 恰好 7 天前, -7: 7 天内, +7: 超过7 天   -amin: 以分钟为单位
        run(""" find . -newer TfTest.py -size -1000000c -atime -7 | xargs ls -lh""")

    def testGit(self):
        print("配置 git 别名, 以树形图的方式显示 log")
        print(""" git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit" """)
        print("git log -p --graph") # 查看具体的 diff 信心, 上面只是修改的树形结构图,但是没有具体 diff 的文件
        print("查看所有的提交信息")
        run("cd /Users/tjuser/Desktop/dataMining; git reflog")
        print("冲突解决工具")
        print("git mergetool")
        print("临时 bug 分支; git stash   暂存工作现场; ... git stash list ; git stash apply stash@{0}进行恢复; git stash drop; ")
        """
        1. 删除暂存区的文件 
        git rm --cached [filename]
        
        2. diff 工作区和暂存区的文件: git diff 
        暂存区 VS 本地仓库: git diff --cached
        工作区 VS 本地仓库: git diff HEAD
        工作区 VS 指定提交: git diff commit-id 
        """



