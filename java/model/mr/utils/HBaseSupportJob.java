package com.rj.recommendation.mr.utils;

import com.alibaba.fastjson.JSON;
import com.google.common.collect.Lists;
import com.mysql.jdbc.StringUtils;
import com.rj.recommendation.mr.algorithms.simrank.WeightedSimrankRecommendation;
import com.rj.recommendation.utils.EncoderUtils;
import com.rj.recommendation.utils.MapUtils;
import com.rj.recommendation.utils.RecUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import com.rj.recommendation.mr.algorithms.simrank.IdIndexJob;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

/**
 * @Author: liuyang103
 * @Creation: 21/01/2018
 */
public class HBaseSupportJob {

    private static final String HBASE_INPUT_PATH = "hbase.simrank.input.path";
    private static final String HBASE_OUTPUT_PATH = "hbase.simrank.output.path";
    private static final String HBASE_COLUMN = "hbase.simrank.column";

    private static final String DEFAULT_HBASE_TABLE_NAME = "hdp_lbg_ectech:qfs_policy_offline";

    private static final byte[] DEFAULT_FAMILY_INDEX = Bytes.toBytes("index");
    private static final String DEFAULT_COLUMN_INDEX = "simrank";

    public static final String NUM_ITEM = com.rj.recommendation.mr.utils.HBaseSupportJob.class.getCanonicalName() + ".item.num";
    public static final String KEY_PREFIX = com.rj.recommendation.mr.utils.HBaseSupportJob.class.getCanonicalName() + ".key.prefix";
    public static final String TOP_K = com.rj.recommendation.mr.utils.HBaseSupportJob.class.getCanonicalName() + ".top.k";
    public static final String ROW_KEY_PREFIX = com.rj.recommendation.mr.utils.HBaseSupportJob.class.getCanonicalName() + ".rowkey.prefix";
    public static final String ITEM_DICTIONARY_PATH = com.rj.recommendation.mr.utils.HBaseSupportJob.class.getCanonicalName() + ".item.dictionary.path";
    public static final String CACHE_FILE = "entid2creatid";

    private enum Counters {
        ID_NUM, Less_Item_Count, Number_Error, No_Creatid_Count, Entity_Format_Error, Query_Save_Count,
        Total_CreatId, Average_Creatid_Count, Multiple_Entity, Truncate_Entity
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = HBaseConfiguration.create(new Configuration());
        String[] dfsArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        Path input = new Path(conf.get(HBASE_INPUT_PATH));
        Path output = new Path(conf.get(HBASE_OUTPUT_PATH));

        Job job = getIdIndexImporterJob(conf, input, output, 200, System.currentTimeMillis());

        if (!job.waitForCompletion(true)) {
            System.out.println("job failed");
            return;
        }
        System.out.println("job success");
    }

    public static Job getIdIndexImporterJob(Configuration conf, Path input, Path output, int numReducer, long ts) throws IOException {
        return getIdIndexImporterJob(conf, input, output, numReducer, ts, DEFAULT_COLUMN_INDEX);
    }

    /**
     * @param conf
     * @param input
     * @param output
     * @param numReducer
     * @param ts         因为会同时有多个任务并行执行，这个可以用来区分HBase中不同任务的rowKey.
     * @param columnName
     * @return
     * @throws IOException
     */
    public static Job getIdIndexImporterJob(Configuration conf, Path input, Path output, int numReducer, long ts, String columnName) throws IOException {

        Configuration hbaseConf = new Configuration(conf);
        HBaseUtils.addHBaseConfiguration(hbaseConf);
        Job job = Job.getInstance(hbaseConf, "IdIndexImporterHBase");
        job.setJarByClass(com.rj.recommendation.mr.utils.HBaseSupportJob.class);

        FileInputFormat.setInputPaths(job, input);
        FileOutputFormat.setOutputPath(job, output);

        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setMapperClass(ImporterMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setReducerClass(ImporterReducer.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);

        job.getConfiguration().set(ROW_KEY_PREFIX, Long.toString(ts));
        job.getConfiguration().set(HBASE_COLUMN, columnName);

        job.setNumReduceTasks(numReducer);
        return job;
    }

    public static Job getDecoderJob(Configuration conf, Path input, Path output, Path cachePath,
                                    int maxItemId, String keyPrefix, int topK, long ts, int numReducer, Path idIndexPath) throws IOException, URISyntaxException {

        FileSystem fs = FileSystem.get(conf);
        Path itemPath = null;
        for (FileStatus status : fs.listStatus(idIndexPath)) {
            if (status.getPath().getName().contains(IdIndexJob.ITEM)) {
                itemPath = status.getPath();
            }
        }
        fs.close();

        if (itemPath == null) {
            System.err.println("no item dictionary path");
            return null;
        }

        Configuration hbaseConf = new Configuration(conf);
        HBaseUtils.addHBaseConfiguration(hbaseConf);
        hbaseConf.set(KEY_PREFIX, keyPrefix);
        hbaseConf.setInt(NUM_ITEM, maxItemId);
        hbaseConf.setInt(TOP_K, topK);

        Job job = Job.getInstance(hbaseConf, "DecoderHBaseJob");
        job.setJarByClass(com.rj.recommendation.mr.utils.HBaseSupportJob.class);

        FileInputFormat.setInputPaths(job, input);
        FileOutputFormat.setOutputPath(job, output);

        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setMapperClass(DecoderMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(VectorWritable.class);

        job.setReducerClass(DecoderReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);

        job.setNumReduceTasks(numReducer);

        job.getConfiguration().set(ROW_KEY_PREFIX, Long.toString(ts));
        job.getConfiguration().set(HBASE_COLUMN, DEFAULT_COLUMN_INDEX);
        job.addCacheFile(new URI(cachePath.toUri() + "#" + CACHE_FILE));
        job.getConfiguration().set(ITEM_DICTIONARY_PATH, itemPath.toString());
        return job;
    }

    private static String getRowKey(String rowKeyPrefix, long index) {
        return EncoderUtils.MD5(rowKeyPrefix + "_" + index);
    }


    private static class ImporterMapper extends Mapper<VarLongWritable, Text, Text, Text> {

        private String rowKeyPrefix;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            rowKeyPrefix = context.getConfiguration().get(ROW_KEY_PREFIX);
        }

        @Override
        protected void map(VarLongWritable key, Text value, Context context) throws IOException, InterruptedException {
            context.write(new Text(getRowKey(rowKeyPrefix, key.get())), value);
        }
    }

    private static class ImporterReducer extends Reducer<Text, Text, LongWritable, Text> {

        private Table table = null;
        private List<Put> puts = null;
        private byte[] column = null;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Connection connection = ConnectionFactory.createConnection(context.getConfiguration());
            table = connection.getTable(TableName.valueOf(DEFAULT_HBASE_TABLE_NAME));
            puts = Lists.newArrayList();
            column = Bytes.toBytes(context.getConfiguration().get(HBASE_COLUMN));
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                Put put = new Put(Bytes.toBytes(key.toString()));
                put.addColumn(DEFAULT_FAMILY_INDEX, column, Bytes.toBytes(value.toString()));
                if (puts.size() == 100) {
                    batchPut(context);
                }
                puts.add(put);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (puts.size() > 0) {
                batchPut(context);
            }
            table.close();
        }

        private void batchPut(Context context) throws IOException {
            table.put(puts);
            context.getCounter(Counters.ID_NUM).increment(puts.size());
            puts.clear();
        }
    }

    private static class DecoderMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

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

            RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE);

            for (Vector.Element element : value.get().nonZeroes()) {
                if (element.index() > maxItemId) {
                    continue;
                }
                vector.set(element.index(), element.get());
            }

            context.write(key, new VectorWritable(vector));
        }
    }

    private static class DecoderReducer extends Reducer<IntWritable, VectorWritable, Text, NullWritable> {

        private Table table = null;
        private int maxItemId = 0;
        private String keyPrefix = null;
        private int topK;
        private String rowKeyPrefix;
        private byte[] column;

        private List<Get> gets;
        private Map<String, String> idTransformMap = null;
        private Map<Integer, String> entIdDictionaryMap = null;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Connection connection = ConnectionFactory.createConnection(context.getConfiguration());
            table = connection.getTable(TableName.valueOf(DEFAULT_HBASE_TABLE_NAME));
            maxItemId = context.getConfiguration().getInt(NUM_ITEM, Integer.MAX_VALUE);
            column = Bytes.toBytes(context.getConfiguration().get(HBASE_COLUMN));
            gets = Lists.newArrayList();

            rowKeyPrefix = context.getConfiguration().get(ROW_KEY_PREFIX);
            keyPrefix = context.getConfiguration().get(KEY_PREFIX);
            topK = context.getConfiguration().getInt(TOP_K, Integer.parseInt(WeightedSimrankRecommendation.DEFAULT_TOP_K));

            idTransformMap = RecUtils.readKeyValue(CACHE_FILE, context.getConfiguration());
            entIdDictionaryMap = new HashMap<>(idTransformMap.size());

            Path itemDictionaryPath = new Path(context.getConfiguration().get(ITEM_DICTIONARY_PATH));

            for (Pair<VarLongWritable, Text> pair : new SequenceFileIterable<VarLongWritable, Text>(itemDictionaryPath, context.getConfiguration())) {
                String itemString = pair.getSecond().toString();
                if (RecUtils.isAdEntity(itemString)) {
                    entIdDictionaryMap.put((int) pair.getFirst().get(), RecUtils.getEntidFromItemId(itemString));
                }
            }

        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            table.close();
        }

        @Override
        protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {

            VectorWritable value = values.iterator().next();

            if (value == null) {
                return;
            }
            process(key, value, context);
        }

        protected void process(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {

            List<Pair<Integer, Double>> orderList = Lists.newArrayList();
            for (Vector.Element element : value.get().nonZeroes()) {

                if (element.index() >= maxItemId) {
                    continue;
                }

                orderList.add(new Pair<Integer, Double>(element.index(), element.get()));
            }

            Collections.sort(orderList, new Comparator<Pair<Integer, Double>>() {
                @Override
                public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
                    return -1 * o1.getSecond().compareTo(o2.getSecond());
                }
            });
/*
            // 这里只保留广告帖子
            Map<String, String> supportMap = getAdResultsByHBase(orderList, context);
//            if (supportMap.size() < topK) {
//                context.getCounter(Counters.Less_Item_Count).increment(1);
//            }

            // use float to save db space.
            Map<String, Float> resultMap = new LinkedHashMap<>();
            for (int i = 0; i < orderList.size(); i++) {
                Pair<Integer, Double> element = orderList.get(i);

                String md5Index = getRowKey(rowKeyPrefix, element.getFirst());
                //String itemString = supportMap.get(md5Index);
                String creatId = supportMap.get(md5Index);
//                if (creatId == null) {
//                    continue;
//                }

                //String entid = null;
//                try {
//                    entid = RecUtils.getEntidFromItemId(itemString);
//                } catch (Exception e) {
//                    context.getCounter(Counters.Entity_Format_Error).increment(1);
//                    continue;
//
//                }
//                String creatId = idTransformMap.get(entid);
                if (creatId == null) {
//                    context.getCounter(Counters.No_Creatid_Count).increment(1);
                    continue;
                }
                resultMap.put(creatId, element.getSecond().floatValue());
            }
*/
            // process key
            Get keyGet = new Get(Bytes.toBytes(getRowKey(rowKeyPrefix, key.get())));
            keyGet.addColumn(DEFAULT_FAMILY_INDEX, column);
            Result result = table.get(keyGet);
            String keyItemString = Bytes.toString(result.getValue(DEFAULT_FAMILY_INDEX, column));
            String keyEntId = RecUtils.getEntidFromItemId(keyItemString);

            Map<String, Float> resultMap = getCreatIdResults(orderList, context);
            if (resultMap.size() < topK * 0.5) {
                context.getCounter(Counters.Less_Item_Count).increment(1);
            }

            context.write(new Text(keyPrefix + keyEntId + RecUtils.FIELDS_DELIMITER + JSON.toJSONString(resultMap)), NullWritable.get());
        }

        private Map<String, Float> getCreatIdResults(List<Pair<Integer, Double>> orderList, Context context) {
            Map<String, Double> results = new HashMap<>(topK);
            for (Pair<Integer, Double> element : orderList) {
                if (entIdDictionaryMap.containsKey(element.getFirst())) {

                    String entId = entIdDictionaryMap.get(element.getFirst());
                    String creatId = idTransformMap.get(entId);
                    if (!StringUtils.isNullOrEmpty(creatId)) {
                        if (results.containsKey(creatId)) {
                            context.getCounter(Counters.Multiple_Entity).increment(1);
                            results.put(creatId, results.get(creatId) + element.getSecond());
                        } else {
                            results.put(creatId, element.getSecond());
                        }
                    }

                }
            }

            LinkedHashMap<String, Float> resultsMap = new LinkedHashMap<>(topK);
            Map<String, Double> sortedMap = MapUtils.sortMapByValue(results, false);
            int index = topK;
            for (String creatId : sortedMap.keySet()) {
                resultsMap.put(creatId, sortedMap.get(creatId).floatValue());
                index--;
                if (index > 0) {
                    continue;
                } else {
                    context.getCounter(Counters.Truncate_Entity).increment(1);
                    break;
                }
            }
            return resultsMap;
        }


        // key is hbase rowKey
        // value is topK creatid
/*        private Map<String, String> getAdResultsByHBase(List<Pair<Integer, Double>> orderList, Context context) throws IOException {
            final int limit = 100;
            List<Get> gets = new ArrayList<>(limit);
            Map<String, String> results = new HashMap<>(orderList.size());

            for (Pair<Integer, Double> element : orderList) {
//                Get get = new Get(Bytes.toBytes(EncoderUtils.MD5(Integer.toString(element.getFirst()))));
                Get get = new Get(Bytes.toBytes(getRowKey(rowKeyPrefix, element.getFirst())));
                get.addColumn(DEFAULT_FAMILY_INDEX, DEFAULT_COLUMN_INDEX);
                gets.add(get);

                if (results.size() >= topK) {
                    context.getCounter(Counters.Query_Save_Count).increment(1);
                    return results;
                }

                if (gets.size() >= limit) {
                    results.putAll(queryHBase(gets));
                }
            }
            if (gets.size() > 0) {
                results.putAll(queryHBase(gets));
            }
            return results;
        }

        private Map<String, String> queryHBase(List<Get> gets) throws IOException {
            Result[] results = table.get(gets);
            Map<String, String> adResults = new HashMap<>();
            gets.clear();
            for (Result result : results) {
                String md5Index = Bytes.toString(result.getRow());
                String itemString = Bytes.toString(result.getValue(DEFAULT_FAMILY_INDEX, DEFAULT_COLUMN_INDEX));
                if (RecUtils.isAdEntity(itemString)) {
                    String creatId = getCreatId(itemString);
                    if (!StringUtils.isNullOrEmpty(creatId)) {
                        adResults.put(md5Index, creatId);
                    }
                }
            }

            return adResults;
        }

        private String getCreatId(String itemString) {
            String entid = RecUtils.getEntidFromItemId(itemString);
            if (entid == null) {
                return null;
            }
            return idTransformMap.get(entid);
        }*/
    }

}
