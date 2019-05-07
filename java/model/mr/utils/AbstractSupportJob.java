package com.rj.recommendation.mr.utils;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapreduce.*;
import org.apache.log4j.Logger;
import org.apache.mahout.common.AbstractJob;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @Author: liuyang103
 * @Creation: 07/12/2017
 */
public abstract class AbstractSupportJob extends AbstractJob implements Closeable {
    private final static Logger logger = Logger.getLogger(AbstractJob.class);

    protected FileSystem fs;
    protected boolean deleteOnExist = false;
    protected boolean keepTmpDir = true;

    protected Path baseInputPath;
    protected Path tmpBaseOutputPath;
    protected Path baseOutputPath;

    /**
     * process hadoop arguments like '-Dmapreduce.job.queuename=root.offline.hdp_lbg_ectech.normal'
     *
     * @param args
     * @return
     */
    public static String[] processArgs(String[] args) {
        List<String> list = new ArrayList<>();
        for (String arg : args) {
            if (arg.startsWith("-D")) {
                continue;
            }
            list.add(arg);
        }
        return list.toArray(new String[list.size()]);
    }

    public AbstractSupportJob() throws IOException {
        super();
        this.fs = FileSystem.get(getConf());
        addInputOption();
        addOutputOption();
        addOption("deleteOnExist", "d", "delete output (and tmp if set) path if exist, default is false", Boolean.FALSE.toString());
        addOption("keepTmpDir", "k", "whether keep tmp directory, default is true", Boolean.TRUE.toString());
    }

    /**
     * This method should be overwrite by subclass and called by the overwrite method.
     *
     * @throws IOException
     */
    public Map<String, List<String>> parseArgs(String[] args) throws IOException {

        Map<String, List<String>> parsedArgs = parseArguments(args);
        if (parsedArgs == null) {
            throw new RuntimeException("No Input Arguments");
        }

        baseInputPath = getInputPath();
        baseOutputPath = getOutputPath();
        // default value is temp
        tmpBaseOutputPath = super.tempPath.getName().equals("temp") ? new Path(baseOutputPath, super.tempPath) : tempPath;
        deleteOnExist = Boolean.valueOf(getOption("deleteOnExist"));
        keepTmpDir = Boolean.valueOf(getOption("keepTmpDir"));


//        if (deleteOnExist) {
//            if (fs.exists(baseOutputPath)) {
//                logger.info("delete output path: " + baseOutputPath.getName());
//                fs.delete(baseOutputPath, true);
//            }
//            if (fs.exists(tmpBaseOutputPath)) {
//                logger.info("delete tmp output path: " + tmpBaseOutputPath.getName());
//                fs.delete(tmpBaseOutputPath, true);
//            }
//        }
        return parsedArgs;
    }

    protected Job prepareJob(Path inputPath,
                             Path outputPath,
                             Class<? extends InputFormat> inputFormat,
                             Class<? extends Mapper> mapper,
                             Class<? extends Writable> mapperKey,
                             Class<? extends Writable> mapperValue,
                             Class<? extends Reducer> reducer,
                             Class<? extends Writable> reducerKey,
                             Class<? extends Writable> reducerValue,
                             Class<? extends OutputFormat> outputFormat, int numberReducer) throws IOException {
        Job job = prepareJob(inputPath, outputPath,
                inputFormat, mapper, mapperKey, mapperValue, reducer, reducerKey, reducerValue, outputFormat);
        job.setNumReduceTasks(numberReducer);
        return job;
    }

    @Override
    public void close() throws IOException {
        if (!isKeepTmpDir() && tmpBaseOutputPath != null && getFileSystem().exists(tmpBaseOutputPath)) {
            deletePath(tmpBaseOutputPath, true);
        }
        fs.close();
    }

    public boolean deletePath(String path, boolean recursive) throws IOException {
        return deletePath(new Path(path), recursive);
    }

    public boolean deletePath(Path path, boolean recursive) throws IOException {
        return fs.delete(path, recursive);
    }

    public FileSystem getFileSystem() {
        return fs;
    }


    public boolean isDeleteOnExist() {
        return deleteOnExist;
    }

    public void setDeleteOnExist(boolean deleteOnExist) {
        this.deleteOnExist = deleteOnExist;
    }

    public boolean isKeepTmpDir() {
        return keepTmpDir;
    }

    public void setKeepTmpDir(boolean keepTmpDir) {
        this.keepTmpDir = keepTmpDir;
    }

    /**
     * 读取HDFS上的压缩文件。
     *
     * @param path
     * @return
     * @throws IOException
     */
    public InputStream getFileStream(Path path)
            throws IOException {
        FileSystem fs = FileSystem.get(getConf());
        CompressionCodecFactory factory = new CompressionCodecFactory(getConf());
        CompressionCodec codec = factory.getCodec(path);
        if (null == codec) {
            return fs.open(path);
        } else {
            return codec.createInputStream(fs.open(path));
        }
    }
}
