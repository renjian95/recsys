package com.rj.recommendation.mr.math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import static com.rj.recommendation.mr.utils.AbstractSupportJob.processArgs;


/**
 * Some operation in {@Code org.apache.mahout.math.hadoop.DistributedRowMatrix} is do not work, probably caused by hadoop version.
 *
 * @Author: renjian01
 * @Creation: 09/12/2017
 */
public class MatrixOperation extends AbstractJob {

    private static final String LEFT = "left";
    private static final String RIGHT = "right";

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new MatrixOperation(), args);
    }

    @Override
    public int run(String[] args) throws Exception {
        addOption("action", "a", "matrix action, transport or multiply", "");
        addOption("numReducer", "numr", "number of reducer,default is 1", "1");

        addInputOption();

        Map<String, List<String>> parsedArgs = parseArguments(processArgs(args));
        if (parsedArgs == null) {
            return -1;
        }
        int numReducer = getInt("numReducer");

        int numRows = Integer.parseInt(getOption("numRows"));
        int numCols = Integer.parseInt(getOption("numCols"));

        DistributedRowMatrix matrix = new DistributedRowMatrix(getInputPath(), getTempPath(), numRows, numCols);
        matrix.setConf(new Configuration(getConf()));

        transposeMatrix(matrix, numReducer, new Path(matrix.getRowPath().getParent(), "transpose_" + (System.nanoTime() & 0xFF)));

        return 0;
    }

    public static DistributedRowMatrix transposeMatrix(DistributedRowMatrix matrix, int numberReducer, Path matrixOutput) throws Exception {
        Path outputPath = new Path(matrixOutput, "transpose-" + (System.nanoTime() & 0xFF));
        Configuration conf = matrix.getConf();

        Job job = HadoopUtil.prepareJob(matrix.getRowPath(), outputPath, SequenceFileInputFormat.class, TransposeMapper.class,
                IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class,
                SequenceFileOutputFormat.class, conf);
        job.setCombinerClass(MergeVectorsReducer.class);
        job.getConfiguration().setInt(TransposeMapper.NEW_NUM_COLS_PARAM, matrix.numRows());
        job.setNumReduceTasks(numberReducer);
        job.setJobName("Transpose Matrix " + matrix.getRowPath().getName());

        if (!job.waitForCompletion(true)) {
            throw new Exception("transposition failed");
        }

        DistributedRowMatrix transposeMatrix = new DistributedRowMatrix(outputPath, new Path(matrixOutput, "tmp"),
                matrix.numCols(), matrix.numRows());
        transposeMatrix.setConf(conf);
        return transposeMatrix;
    }

    public static DistributedRowMatrix matrixMultiply(Path leftPath, int leftRowNum, int leftColumnNum,
                                                      Path rightPath, int rightRowNum, int rightColumnNum, Path temp,
                                                      int numReducer, Path resultPath, Configuration conf) throws Exception {

        DistributedRowMatrix leftMatrix = new DistributedRowMatrix(leftPath, new Path(temp, "left_" + (System.nanoTime() & 0xFF)), leftRowNum, leftColumnNum);
        DistributedRowMatrix rightMatrix = new DistributedRowMatrix(rightPath, new Path(temp, "right_" + (System.nanoTime() & 0xFF)), rightRowNum, rightColumnNum);

        leftMatrix.setConf(conf);
        rightMatrix.setConf(conf);

        return matrixMultiply(leftMatrix, rightMatrix, numReducer, resultPath, temp);
    }


    /**
     * @param leftMatrix
     * @param rightMatrix
     * @param numberReducer
     * @param matrixOutput
     * @param tempOutput
     * @return
     * @throws Exception
     */
    public static DistributedRowMatrix matrixMultiply(DistributedRowMatrix leftMatrix, DistributedRowMatrix rightMatrix,
                                                      int numberReducer, Path matrixOutput, Path tempOutput) throws Exception {

        if (leftMatrix.numRows() != rightMatrix.numCols()) {
            throw new CardinalityException(leftMatrix.numRows(), rightMatrix.numCols());
        }

        Configuration conf = new Configuration(leftMatrix.getConf());
        Path job1Output = new Path(leftMatrix.getOutputTempPath(), "matrix_multiply" + (System.nanoTime() & 0xFF));

        Job job1 = Job.getInstance(conf, "Matrix Multiply stage1:" + leftMatrix.getRowPath().getName() + "-" + rightMatrix.getRowPath().getName());
        job1.setJarByClass(MatrixOperation.class);
        MultipleInputs.addInputPath(job1, leftMatrix.getRowPath(), SequenceFileInputFormat.class, LeftMatrixMultiplyMapper.class);
        MultipleInputs.addInputPath(job1, rightMatrix.getRowPath(), SequenceFileInputFormat.class, RightMatrixMultiplyMapper.class);
        job1.setReducerClass(MatrixMultiplyReducer.class);
        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(VectorWritable.class);
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(VectorWritable.class);
        job1.setOutputFormatClass(SequenceFileOutputFormat.class);
        job1.setNumReduceTasks(numberReducer);
        FileOutputFormat.setOutputPath(job1, job1Output);

        if (!job1.waitForCompletion(true)) {
            throw new Exception(job1.getJobName() + "matrix multiply job failed");
        }

        Job job2 = HadoopUtil.prepareJob(job1Output, matrixOutput, SequenceFileInputFormat.class, MatrixMultiplyAggregateMapper.class, IntWritable.class, VectorWritable.class,
                MatrixMultiplyAggregateReducer.class, IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class, conf);
        job2.setNumReduceTasks(numberReducer);
        job2.setJobName("Matrix Multiply stage2" + leftMatrix.getRowPath().getName() + "-" + rightMatrix.getRowPath().getName());
        if (!job2.waitForCompletion(true)) {
            throw new Exception(job1.getJobName() + "matrix multiply job failed");
        }

        DistributedRowMatrix matrix = new DistributedRowMatrix(matrixOutput, new Path(tempOutput, "mutiply_tmp_" + (System.nanoTime() & 0xFF)),
                leftMatrix.numRows(), rightMatrix.numCols());
        matrix.setConf(conf);
        return matrix;
    }

    private static class LeftMatrixMultiplyMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {


        @Override
        protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
            Vector vector = value.get();

            for (Vector.Element element : vector.nonZeroes()) {
                Vector columnVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
                // left matrix decide the row index of the output matrix element.
                columnVector.set(key.get(), element.get());
                NamedVector namedVector = new NamedVector(columnVector, LEFT);
                context.write(new IntWritable(element.index()), new VectorWritable(namedVector));
            }
        }
    }

    private static class RightMatrixMultiplyMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
            Vector vector = value.get();
            NamedVector namedVector = new NamedVector(vector, RIGHT);
            context.write(key, new VectorWritable(namedVector));
        }
    }

    private static class MatrixMultiplyReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
            List<Vector.Element> lefts = new ArrayList<>(1000);
            Vector right = null;

            for (VectorWritable value : values) {
                NamedVector tmp = (NamedVector) value.get();
                if (tmp.getName().equals(LEFT)) {
                    for (Vector.Element element : tmp.getDelegate().nonZeroes()) {
                        lefts.add(element);
                    }
                } else if (tmp.getName().equals(RIGHT)) {
                    right = tmp.getDelegate();
                }
            }

            if (lefts.size() == 0 || right == null) {
                return;
            }

            for (Vector.Element left : lefts) {
                int rowIndex = left.index();
                context.write(new IntWritable(rowIndex), new VectorWritable(right.times(left.get())));
            }
        }
    }

    private static class MatrixMultiplyAggregateMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    private static class MatrixMultiplyAggregateReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
            Iterator<VectorWritable> it = values.iterator();
            if (!it.hasNext()) {
                return;
            }
            Vector accumulator = new RandomAccessSparseVector(it.next().get());
            while (it.hasNext()) {
                Vector row = it.next().get();
                accumulator.assign(row, Functions.PLUS);
            }
            context.write(key, new VectorWritable(accumulator));
        }
    }


}
