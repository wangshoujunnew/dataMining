# LNMP 一键安装 https://www.jianshu.com/p/ad530e10a089
1. wget http://soft.vpser.net/lnmp/lnmp1.7.tar.gz -cO lnmp1.7.tar.gz 
2. tar zxf lnmp1.7.tar.gz 
3. cd lnmp1.7  
4. ./install.sh lnmp

# 端口转发 https://www.cnblogs.com/kevingrace/p/9453987.html
容器除了在启动时添加端口映射关系，还可以通过宿主机的iptables进行nat转发，将宿主机的端口映射到容器的内部端口上，这种方式适用于容器启动时没有指定端口映射的情况！
服务 /home/wwwroot/default
各个服务的重新启动

# docker 去部署 LNMP

## docker 使用, 更换阿里元进项 https://blog.csdn.net/tengxing007/article/details/103071957
1. 启动 docker的守护进程
sudo service docker start
2. 查找镜像, 镜像的全称是 <username>/<repository>
docker search ubuntu
3. 下载
docker pull learn/tutorial
4. 运行 docker 容器 
docker run --name=xxx -dit ubuntu -p docker端口:localPort # -it 建立虚拟终端, 进入,exec进入, run 启动
docker exec -it name /bin/bash
停止: sudo docker stop brave_colden
移除: docker rm brave_colden
5. 所有容器
docker ps -a
6. 所有进项
docker image ls 
7. 本地和 docker 文件传送
docker cp localF dockerName:path


# docker file 创建
```shell script
FROM centos:7

ENV PYTHON_VERSION "3.6.5"

RUN yum install -y \
    wget \
    gcc make \
    zlib-dev openssl-devel sqlite-devel bzip2-devel

COPY yum.repos.d/* /etc/yum.repos.d/
COPY Python-3.6.5.tgz .


RUN tar xvf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --prefix=/usr/local \
    && make \
    && make altinstall \
    && cd / \
    && rm -rf Python-${PYTHON_VERSION}*

ENV PATH "/usr/local/bin:${PATH}"

RUN echo -e "y\n" | yum install mariadb-devel

RUN echo -e "y\ny\n" | yum install openssl

RUN echo -e "y\ny\n" | yum install git
```


-- 真实表创建
CREATE TABLE `tujia_citylandmark_u2o` (
`date` char(8) NOT NULL COMMENT '日期',
`scene` char(8) NOT NULL COMMENT '场景',
`bucket` char(8) NOT NULL COMMENT '桶',
`homepagepv` int(10) DEFAULT '0' COMMENT '首页pv',
`homepageuv` int(10) DEFAULT '0' COMMENT '首页uv',
`listpagepv` int(10) DEFAULT '0' COMMENT '列表页pv',
`listpageuv` int(10) DEFAULT '0' COMMENT '列表页uv',
`detailpagepv` int(10) DEFAULT '0' COMMENT '详情页pv',
`detailpageuv` int(10) DEFAULT '0' COMMENT '详情页uv',
`bookpagepv` int(10) DEFAULT '0' COMMENT 'book页pv',
`bookpageuv` int(10) DEFAULT '0' COMMENT 'book页uv',
`orderpagepv` int(10) DEFAULT '0' COMMENT 'order页pv',
`orderpageuv` int(10) DEFAULT '0' COMMENT 'order页uv',
`unit_showall` int(10) DEFAULT NULL COMMENT '房屋曝光pv',
`unit_showuniq` int(10) DEFAULT NULL COMMENT '曝光房屋数'
) ENGINE=MyISAM DEFAULT CHARSET=utf8 COMMENT='城市地标监控表u2o监控表'

-- 导出数据成 sql
mysqldump -u root -p RUNOOB runoob_tbl > dump.txt


# 数据统计维度
1. 日,周,月变化
2. 周几
3. 过去 7, 15, 30, 60, 120, 180 天