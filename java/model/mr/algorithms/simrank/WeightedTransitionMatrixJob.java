package com.rj.recommendation.mr.algorithms.simrank;

import com.rj.recommendation.utils.RecUtils;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

/**
 * row is item, column is user.
 * <p>
 * input format: userId, itemId, score, userIdLong, itemIdLong
 * i.e. 007a3d46cbe6f8868ec066f32cb1568923a9ee75	j_143_18522199851528	0.5	3222913	54563
 *
 * @Author: renjian01
 * @Creation: 07/12/2017
 */
public class WeightedTransitionMatrixJob {

    public enum VarianceCounters {
        DIMENSION, NAN_MAPPER, NAN_REDUCER
    }

    public static final String DECAY_FACTOR = WeightedTransitionMatrixJob.class.getCanonicalName() + ".decay.factor";
    public static final String UNIT_MATRIX_OUTPUT_PATH = WeightedTransitionMatrixJob.class.getCanonicalName() + ".unit.output.path";
    public static final String DECAY_MATRIX_OUTPUT_PATH = WeightedTransitionMatrixJob.class.getCanonicalName() + ".decay.output.path";


    public static class WeightedTransitionMatrixMapper extends Mapper<LongWritable, Text, VarLongWritable, EntityPrefWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length < 5) {
                return;
            }
            try {
                double score = Double.parseDouble(tokens[2]);
                long userIdLong = Long.parseLong(tokens[3]);
                long itemIdLong = Long.parseLong(tokens[4]);
                context.write(new VarLongWritable(userIdLong), new EntityPrefWritable(itemIdLong, (float) score));
                context.write(new VarLongWritable(itemIdLong), new EntityPrefWritable(userIdLong, (float) score));
            } catch (NumberFormatException e) {
                context.getCounter(VarianceCounters.NAN_MAPPER).increment(1);
                return;
            }

        }
    }

    public static class WeightedTransitionMatrixReducer extends Reducer<VarLongWritable, EntityPrefWritable, IntWritable, VectorWritable> {

        private MultipleOutputs multipleOutputs;
        private String unitOutput;
        private String decayOutput;
        private double decayFactor;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            multipleOutputs = new MultipleOutputs(context);
            unitOutput = context.getConfiguration().get(UNIT_MATRIX_OUTPUT_PATH);
            decayOutput = context.getConfiguration().get(DECAY_MATRIX_OUTPUT_PATH);
            decayFactor = context.getConfiguration().getDouble(DECAY_FACTOR, Double.parseDouble(WeightedSimrankRecommendation.DEFAULT_DECAY_FACTOR));
        }

        @Override
        protected void reduce(VarLongWritable key, Iterable<EntityPrefWritable> values, Context context) throws IOException, InterruptedException {
            Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 20);
            SummaryStatistics stats = new SummaryStatistics();

            for (EntityPrefWritable value : values) {
                stats.addValue(value.getPrefValue());
                // TODO risk here
                vector.set((int) value.getID(), value.getPrefValue());
            }
            double variance = stats.getVariance();
            double sum = stats.getSum();
            if (variance == Double.NaN) {
                context.getCounter(VarianceCounters.NAN_REDUCER).increment(1);
                // TODO to be improve
                variance = 0;
            }
            context.getCounter(VarianceCounters.DIMENSION).increment(1);

            double spread = Math.exp(variance * -1);
            double totalP = 0;
            // vector.divide(stats.getSum()) is not efficient
            for (Vector.Element element : vector.nonZeroes()) {
                element.set(element.get() / sum * spread);
                totalP += element.get();
            }

            // the transition probability to 1itself
            int rowId = (int) key.get();
            vector.set(rowId, 1 - totalP);
            context.write(new IntWritable(rowId), new VectorWritable(vector, false));

            // unit and decay vector
            Vector unitVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
            unitVector.set(rowId, 1.0);
            multipleOutputs.write(new IntWritable(rowId), new VectorWritable(unitVector, false), unitOutput);

            Vector decayVector = unitVector.like();
            decayVector.set(rowId, decayFactor);
            multipleOutputs.write(new IntWritable(rowId), new VectorWritable(decayVector, false), decayOutput);

        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            multipleOutputs.close();
        }
    }


}
