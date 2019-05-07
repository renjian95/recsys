package com.rj.recommendation.mr.algorithms.simrank;

import com.rj.recommendation.mr.math.MatrixOperation;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import java.io.IOException;

/**
 * @Author: liuyang103
 * @Creation: 08/12/2017
 */
public class RandomWalkJob {

    final private DistributedRowMatrix similarityMatrix;
    final private DistributedRowMatrix decayMatrix;
    final private DistributedRowMatrix weightedTransitionMatrix;

    final private Path tmpBasePath;
    final private Path similarityOutputPath;
    final private boolean isFirstIteration;
    final private int dimension;
    final private int numReducer;

    public RandomWalkJob(DistributedRowMatrix similarityMatrix, DistributedRowMatrix decayMatrix,
                         DistributedRowMatrix weightedTransitionMatrix, Path tmpBasePath, Path similarityOutputPath,
                         boolean isFirstIteration, int dimension, int numberReducer) {
        this.similarityMatrix = similarityMatrix;
        this.decayMatrix = decayMatrix;
        this.weightedTransitionMatrix = weightedTransitionMatrix;
        this.tmpBasePath = tmpBasePath;
        this.isFirstIteration = isFirstIteration;
        this.dimension = dimension;
        this.numReducer = numberReducer;
        this.similarityOutputPath = similarityOutputPath;
    }

    public DistributedRowMatrix iterateSimilarityMatrix(Configuration conf) throws Exception {

        Path tmpPath = new Path(tmpBasePath, "tmp");
        // decay matrix is diagonal matrix, no need to transpose this matrix.
        DistributedRowMatrix part1 = MatrixOperation.matrixMultiply(decayMatrix, weightedTransitionMatrix, numReducer,
                new Path(tmpBasePath, "part1"), tmpPath);

        DistributedRowMatrix part2;
        // a trick here, if first iteration, similarity matrix is unit matrix.
        if (isFirstIteration) {
            part2 = part1;
        } else {
            part2 = MatrixOperation.matrixMultiply(part1, similarityMatrix, numReducer, new Path(tmpPath, "part2"), tmpPath);
        }

        DistributedRowMatrix tmpMatrix = MatrixOperation.matrixMultiply(part2, weightedTransitionMatrix, numReducer,
                new Path(tmpBasePath, "part3"), tmpPath);

        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(similarityOutputPath)) {
            fs.delete(similarityOutputPath, true);
        }

        Job iterJob = Job.getInstance(conf, "RandomWalk" + tmpBasePath.getName());
        iterJob.setJarByClass(RandomWalkJob.class);
        iterJob.setInputFormatClass(SequenceFileInputFormat.class);
        iterJob.setOutputFormatClass(SequenceFileOutputFormat.class);
        iterJob.setMapperClass(TempMapper.class);
        iterJob.setOutputKeyClass(IntWritable.class);
        iterJob.setOutputValueClass(VectorWritable.class);
        iterJob.setNumReduceTasks(0);
        FileInputFormat.addInputPath(iterJob, tmpMatrix.getRowPath());
        FileOutputFormat.setOutputPath(iterJob, similarityOutputPath);

        if (!iterJob.waitForCompletion(true)) {
            throw new Exception(iterJob.getJobName() + " failed.");
        }

        DistributedRowMatrix result = new DistributedRowMatrix(similarityOutputPath, tmpPath, dimension, dimension);
        result.setConf(conf);
        return result;
    }

    public static class TempMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
            int rowId = key.get();
            Vector vector = value.get();
            Vector outVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
            for (Vector.Element element : vector.nonZeroes()) {
                int columnId = element.index();
                double elementValue = element.get();
                if (rowId == columnId) {
                    elementValue = 1;
                }
                outVector.set(columnId, elementValue);
            }
            context.write(key, new VectorWritable(outVector));
        }
    }


}
