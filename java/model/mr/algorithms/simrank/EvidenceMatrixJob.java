package com.rj.recommendation.mr.algorithms.simrank;


import com.rj.recommendation.utils.RecUtils;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @Author: renjian01
 * @Creation: 09/12/2017
 */
public class EvidenceMatrixJob {

    private static final IntWritable ONE = new IntWritable(1);

    public static class CooccurrenceMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length < 5) {
                return;
            }
            try {
                int userId = Integer.parseInt(tokens[3]);
                int itemId = Integer.parseInt(tokens[4]);

                context.write(new IntWritable(userId), new IntWritable(itemId));
                context.write(new IntWritable(itemId), new IntWritable(userId));
            } catch (NumberFormatException e) {
                e.printStackTrace();
            }

        }
    }

    public static class CooccurrenceReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

            List<Integer> list = new ArrayList<>(200);
            for (IntWritable value : values) {
                list.add(value.get());
            }

            for (Integer outer : list) {
                for (Integer inner : list) {
                    if (outer == inner) {
                        continue;
                    }
                    context.write(new IntWritable(outer), new IntWritable(inner));
                }
            }

        }
    }

    public static class EvidenceMapper extends Mapper<IntWritable, IntWritable, Text, IntWritable> {

        @Override
        protected void map(IntWritable key, IntWritable value, Context context) throws IOException, InterruptedException {
            context.write(new Text(key.get() + RecUtils.FIELDS_DELIMITER + value.get()), ONE);
        }
    }

    public static class EvidenceReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

            String[] tokens = RecUtils.splitPrefTokens(key.toString());
            if (tokens.length < 2) {
                return;
            }

            int id1 = Integer.parseInt(tokens[0]);
            int id2 = Integer.parseInt(tokens[1]);

            int cooccurrenceCount = 0;
            for (IntWritable value : values) {
                cooccurrenceCount++;
            }

            double weight = 0;
            for (int i = 1; i <= cooccurrenceCount; i++) {
                weight += Math.pow(2, -i);
            }

            context.write(key, new DoubleWritable(weight));
        }
    }

    public static class EvidenceMatrixMapper extends Mapper<Text, DoubleWritable, IntWritable, VectorWritable> {

        @Override
        protected void map(Text key, DoubleWritable value, Context context) throws IOException, InterruptedException {

            String[] tokens = RecUtils.splitPrefTokens(key.toString());
            if (tokens.length < 2) {
                return;
            }

            int id1 = Integer.parseInt(tokens[0]);
            int id2 = Integer.parseInt(tokens[1]);

            Vector vector1 = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
            vector1.set(id2, value.get());

            Vector vector2 = vector1.like();
            vector2.set(id1, value.get());

            context.write(new IntWritable(id1), new VectorWritable(vector1));
            context.write(new IntWritable(id2), new VectorWritable(vector2));
        }
    }

    public static class EvidenceMatrixReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        @Override
        protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {

            Iterator<VectorWritable> iterator = values.iterator();
            if (!iterator.hasNext()) {
                return;
            }

            Vector accumulator = new RandomAccessSparseVector(iterator.next().get());
            while (iterator.hasNext()) {
                Vector vector = iterator.next().get();

                for (Vector.Element element : vector.nonZeroes()) {
                    accumulator.set(element.index(), element.get());
                }
            }

            context.write(key, new VectorWritable(accumulator));

        }
    }

}
