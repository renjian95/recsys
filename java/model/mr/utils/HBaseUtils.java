package com.rj.recommendation.mr.utils;

import org.apache.hadoop.conf.Configuration;

/**
 * @Author: renjian01
 * @Creation: 01/04/2018
 */
public class HBaseUtils {


    /**
     * 对于离线表，线上环境默认为正确，可不用额外指定；但在线表需要额外指定
     *
     * @param conf
     * @return
     */
    public static Configuration addHBaseConfiguration(Configuration conf) {
        conf.set("hbase.zookeeper.quorum", "10.126.81.134:2181,10.126.81.135:2181,10.126.81.136:2181,10.126.81.224:2181,10.126.81.225:2181");
        conf.set("hbase.rootdir", "hdfs://hbase-58-cluster/home/hbase");
        return conf;
    }
}
