package com.rj.recommendation.mr.algorithms.simrank;

import com.rj.recommendation.utils.RecUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VarLongWritable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Two Stage (use Hadoop configuration):
 * <p>
 * Stage One, append userId (in Long) to the "userId,itemId,score" record.
 * <p>
 * Stage Two, append itemId (in Long) to the "userId, itemId,score" record.
 * <p>
 * Finally output: userId, itemId, score, userIdLong, itemIdLong
 * <p>
 *
 * @Author: liuyang103
 *
 * @Creation: 07/12/2017
 */
public class IdDataJob {

    public static final String INDEX = IdDataJob.class.getCanonicalName() + ".index";

    public static final int USER_INDEX = 0;
    public static final int ITEM_INDEX = 1;

    public static class IdDataMapper extends Mapper<LongWritable, Text, Text, Text> {

        // use to choose userId index or itemId index.
        private int index = USER_INDEX;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            index = context.getConfiguration().getInt(INDEX, USER_INDEX);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            if (value == null || value.toString().length() == 0) {
                return;
            }
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length < 3) {
                return;
            }
            context.write(new Text(tokens[index]), value);
        }
    }

    public static class IdIndexMapper extends Mapper<VarLongWritable, Text, Text, Text> {
        @Override
        protected void map(VarLongWritable key, Text value, Context context) throws IOException, InterruptedException {
            context.write(value, new Text(key.toString()));
        }
    }

    public static class IdDataReducer extends Reducer<Text, Text, Text, NullWritable> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Long idLong = null;
            List<String> list = new ArrayList<>();
            // TODO improve code, use second sort instead of keep in memory.
            for (Text value : values) {
                String[] tokens = RecUtils.splitPrefTokens(value.toString());

                if (tokens.length == 1) {
                    try {
                        idLong = Long.parseLong(tokens[0]);
                    } catch (NumberFormatException e) {
                        return;
                    }
                } else {
                    list.add(value.toString());
                }

            }

            for (String s : list) {
                context.write(new Text(s + RecUtils.FIELDS_DELIMITER + idLong.toString()), NullWritable.get());
            }

        }
    }

}
