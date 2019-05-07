package com.rj.recommendation.mr.algorithms.simrank;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;

/**
 * @Author: liuyang103
 * @Creation: 11/12/2017
 */
public class DistributedCacheUtil {

    public static final Log LOG = LogFactory.getLog(DistributedCacheUtil.class);

    public DistributedCacheUtil() {
    }

    public static void cpFile(Path sourceFilePath, Path destinationDirectory) throws IOException {
        resultExeCmd("mv " + sourceFilePath.toString() + " " + destinationDirectory.toString());
    }

    public static void cpFile(String sourceFilePath, String destinationDirectory) throws IOException {
        resultExeCmd("mv " + sourceFilePath + " " + destinationDirectory);
    }

    public static String getFileAbsolutePath(Path path, Configuration conf, boolean isLocal) throws IOException {
        String resolvedPath = resultExeCmd("readlink -f " + path.toString());
        return resolvedPath.trim();
    }

    public static InputStream getFileStream(Path path, Configuration conf, boolean isLocal) throws IOException {
        FileSystem fs = null;
        if (!isLocal) {
            fs = FileSystem.get(conf);
        } else {
            String resolvedPath = resultExeCmd("readlink -f " + path.toString());
            path = new Path(resolvedPath.trim());
            fs = FileSystem.get(URI.create("file:///"), conf);
        }

        CompressionCodecFactory factory = new CompressionCodecFactory(conf);
        CompressionCodec codec = factory.getCodec(path);
        LOG.info("path=" + path + ", codec=" + codec);
        return (InputStream) (null == codec ? fs.open(path) : codec.createInputStream(fs.open(path)));
    }

    public static String resultExeCmd(String cmd) throws IOException {
        String returnString = "";
        Process pro = null;
        Runtime runTime = Runtime.getRuntime();
        if (runTime == null) {
            throw new IOException("Create runtime false!");
        } else {
            BufferedReader input = null;

            try {
                pro = runTime.exec(cmd);

                String line;
                for (input = new BufferedReader(new InputStreamReader(pro.getInputStream())); (line = input.readLine()) != null; returnString = returnString + line + "\n") {
                    ;
                }
            } catch (IOException var9) {
                throw new IOException("execute command failed. cmd=" + cmd);
            } finally {
                if (input != null) {
                    input.close();
                }

                if (pro != null) {
                    pro.destroy();
                }

            }

            return returnString;
        }
    }

}
