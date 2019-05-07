package com.rj.recommendation.mr.algorithms.simrank;

import com.rj.recommendation.utils.RecUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;

/**
 * Two Stage.
 * Stage One, do id`s distinct job.
 * Stage two, index id.
 *
 * @Author: liuyang103
 * @Creation: 07/12/2017
 */
public class IdIndexJob {

    public static final String ITEM = "item";
    public static final String USER = "user";

    public enum Counters {
        NUM_USER, NUM_ITEM
    }

    public static int getStage2NumberReducer() {
        return 1;
    }

    public static class IdIndexMapper1 extends Mapper<LongWritable, Text, Text, Text> {


        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            if (value == null || value.toString().length() == 0) {
                return;
            }
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length < 2) {
                return;
            }
            context.write(new Text(tokens[0]), new Text(USER));
            context.write(new Text(tokens[1]), new Text(ITEM));

        }
    }

    public static class IdIndexReducer1 extends Reducer<Text, Text, Text, NullWritable> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(new Text(key.toString() + RecUtils.FIELDS_DELIMITER + value.toString()), NullWritable.get());
                return;
            }

        }
    }


    public static class IdIndexMapper2 extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // key  userId/itemId \t tag
            // i.e.
            // ++I9fViha3jZmPbTTXoiRQ	user
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length != 2) {
                return;
            }
            context.write(new Text(tokens[1]), new Text(tokens[0]));
        }
    }


    public static class IdIndexReducer2 extends Reducer<Text, Text, VarLongWritable, Text> {
        private final VarLongWritable idLongWritable = new VarLongWritable();

        private MultipleOutputs multipleOutputs;
        private long index = 0l;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            multipleOutputs = new MultipleOutputs(context);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String tag = key.toString();
            for (Text value : values) {
                idLongWritable.set(index++);
                if (tag.equals(USER)) {
                    context.getCounter(Counters.NUM_USER).increment(1);
                }
                if (tag.equals(ITEM)) {
                    context.getCounter(Counters.NUM_ITEM).increment(1);
                }
                multipleOutputs.write(key.toString(), idLongWritable, value);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            super.cleanup(context);
            multipleOutputs.close();
        }
    }
}
