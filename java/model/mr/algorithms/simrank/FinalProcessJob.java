package com.rj.recommendation.mr.algorithms.simrank;


import com.alibaba.fastjson.JSON;
import com.rj.recommendation.utils.RecUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.*;

/**
 * stage 1: join to filter user, only left item.
 *
 * @Author: renjian01
 * @Creation: 09/12/2017
 */
public class FinalProcessJob {

    public static final String CACHE_FILE = "entid2creatid";
    public static final String KEY_PREFIX = FinalProcessJob.class.getCanonicalName() + ".key.prefix";
    public static final String TOP_K = FinalProcessJob.class.getCanonicalName() + ".top.k";

    public static final String NUM_ITEM = FinalProcessJob.class.getCanonicalName() + ".item.num";

    public static final String INDEX = FinalProcessJob.class.getCanonicalName() + ".index";

    public static final int LEFT = 0;
    public static final int RIGHT = 1;

    public static class SimilarityMatrixFilterMapper extends Mapper<IntWritable, VectorWritable, IntWritable, Text> {

        private int maxItemId = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            maxItemId = context.getConfiguration().getInt(NUM_ITEM, Integer.MAX_VALUE);
        }

        @Override
        protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
            int id = key.get();
            if (id >= maxItemId) {
                // we only consider item similarity.
                return;
            }
            for (Vector.Element element : value.get().nonZeroes()) {
                if (element.index() >= maxItemId) {
                    continue;
                }

                context.write(new IntWritable(id), new Text(id + RecUtils.FIELDS_DELIMITER + element.index()
                        + RecUtils.FIELDS_DELIMITER + element.get()));
            }


        }
    }

    public static class IdIndexReaderMapper extends Mapper<VarLongWritable, Text, IntWritable, Text> {
        private int maxItemId = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            maxItemId = context.getConfiguration().getInt(NUM_ITEM, Integer.MAX_VALUE);
        }

        @Override
        protected void map(VarLongWritable key, Text value, Context context) throws IOException, InterruptedException {
            int id = (int) key.get();
            if (id >= maxItemId) {
                return;
            }
            context.write(new IntWritable(id), value);
        }
    }


    public static class SimilarityReaderMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

        private int index = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            index = context.getConfiguration().getInt(INDEX, LEFT);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = RecUtils.splitPrefTokens(value.toString());

            if (tokens.length < 3) {
                return;
            }

            try {
                int id = Integer.parseInt(tokens[index]);
                context.write(new IntWritable(id), value);
            } catch (NumberFormatException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }
    }

    public static class JoinReducer extends Reducer<IntWritable, Text, NullWritable, Text> {

        private int index = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            index = context.getConfiguration().getInt(INDEX, LEFT);
        }

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String stringId = null;
            List<String> list = new ArrayList<>();
            for (Text value : values) {

                String[] tokens = RecUtils.splitPrefTokens(value.toString());
                if (tokens.length == 1) {
                    stringId = value.toString();
                } else {
                    list.add(value.toString());
                }
            }

            for (String value : list) {

                String[] tokens = RecUtils.splitPrefTokens(value.toString());
                if (tokens.length < index) {
                    return;
                }
                tokens[index] = stringId;
                StringBuilder builder = new StringBuilder();

                // append to a string
                for (String token : tokens) {
                    if (builder.length() > 0) {
                        builder.append(RecUtils.FIELDS_DELIMITER);
                    }
                    builder.append(token);
                }

                context.write(NullWritable.get(), new Text(builder.toString()));

            }

        }
    }


    public static class FilterAndSortMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = RecUtils.splitPrefTokens(value.toString());
            if (tokens.length < 3) {
                return;
            }

            if (RecUtils.isAdEntity(tokens[0]) || RecUtils.isAdEntity(tokens[1])) {
                context.write(new Text(RecUtils.getEntidFromItemId(tokens[0])),
                        new Text(RecUtils.getEntidFromItemId(tokens[1]) + RecUtils.FIELDS_DELIMITER + tokens[2]));
            }
        }
    }

    public static class FilterAndSortReducer extends Reducer<Text, Text, Text, NullWritable> {

        private Map<String, String> idTransformMap = null;
        private String keyPrefix = null;
        private int topK;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            idTransformMap = RecUtils.readKeyValue(CACHE_FILE, context.getConfiguration());
            keyPrefix = context.getConfiguration().get(KEY_PREFIX);
            topK = context.getConfiguration().getInt(TOP_K, Integer.parseInt(WeightedSimrankRecommendation.DEFAULT_TOP_K));
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String entIdKey = keyPrefix + key.toString();

            Map<String, Pair<String, Double>> map = new HashMap<>();
            for (Text value : values) {
                String[] parts = RecUtils.splitPrefTokens(value.toString());

                if (parts.length < 2) {
                    return;
                }

                double weight = Double.parseDouble(parts[1]);

                String creatIdValue = idTransformMap.get(parts[0]);
                if (!RecUtils.isLegalCreatId(creatIdValue)) {
                    continue;
                }


                if (map.containsKey(creatIdValue)) {
                    Pair<String, Double> tmp = map.get(creatIdValue);
                    Pair<String, Double> newTmp = Pair.of(creatIdValue, tmp.getSecond() + weight);
                    map.put(creatIdValue, newTmp);

                } else {
                    map.put(creatIdValue, Pair.of(creatIdValue, weight));
                }
            }

            List<Pair<String, Double>> list = new ArrayList<>(map.values());
            // sort by weight
            Collections.sort(list, new Comparator<Pair<String, Double>>() {

                @Override
                public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
                    return -1 * (o1.getSecond().compareTo(o2.getSecond()));
                }
            });

            Map<String, Float> resultMap = new LinkedHashMap<>();
            int index = topK;
            for (Pair<String, Double> pair : list) {
                resultMap.put(pair.getFirst(), pair.getSecond().floatValue());
                index--;
                if (index > 0) {
                    continue;
                } else {
                    break;
                }
            }

            context.write(new Text(entIdKey + RecUtils.FIELDS_DELIMITER + JSON.toJSONString(resultMap)), NullWritable.get());

        }

    }


}
