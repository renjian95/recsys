package com.rj.recommendation.mr.algorithms.simrank;




import com.rj.recommendation.mr.algorithms.simrank.IdDataJob;
import com.rj.recommendation.mr.algorithms.simrank.RandomWalkJob;
import com.rj.recommendation.mr.math.MatrixOperation;
import com.rj.recommendation.mr.utils.AbstractSupportJob;
import com.rj.recommendation.mr.utils.HBaseSupportJob;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The input of this job should be in format of "user,item,score" or "user \t item \t score".
 * <p>
 * i.e.
 * <p>
 * +LIVelowxZnQ/sf8WPVQSA	p_8703_30370148922824	0.71428
 * <p>
 * User or item entity can be arbitrary value, i.e. numeral or alphabet, because this recommendation will do the encode job.
 * <p>
 * <b>Notice: Phase 9 is relay on this item format.</b>
 *
 * @Author: liuyang103
 * @Creation: 07/12/2017
 */
public class WeightedSimrankRecommendation extends AbstractSupportJob {

    public static final String ID_DICTIONARY_PATH = "id_dictionary";
    public static final String USER_ITEM_DATA_PATH = "user_item_data";

    public static final String DEFAULT_DECAY_FACTOR = "0.8";
    public static final String DEFAULT_TOP_K = "30";

    /**
     * the argument (-Dmapreduce.job.queuename=root.offline.hdp_lbg_ectech.normal) should be passed to the
     * {@code ToolRunner}, so method {@code processArgs} should not be called in ToolRunner.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        try (WeightedSimrankRecommendation operation = new WeightedSimrankRecommendation()) {
            ToolRunner.run(operation, args);
        }
    }

    public WeightedSimrankRecommendation() throws IOException {
        super();
    }

    @Override
    public Map<String, List<String>> parseArgs(String[] args) throws IOException {
        return super.parseArgs(args);
    }


    @Override
    public int run(String[] args) throws Exception {
        addOption("numberReducer", "nd", "number of reducer, default is 200", "200");
        addOption("decayFactor", "decay", "decay factor, default is 0.8", DEFAULT_DECAY_FACTOR);
        addOption("numIteration", "niter", "number of random walk iteration, default is 7", "7");
        addOption("topKRecommendation", "top", "number of ads to recommendation, default is 30", DEFAULT_TOP_K);
        addOption("idTransformPath", null, "the path of idTransform, used to transform entid to creatid", true);
        addOption("keyPrefix", null, "the prefix of key in the final output", true);
        addOption("useHBase", null, "if true, use hbase to store the itemId and userId, default is false", "false");

        addOption("useEvidence", null, "if true, use evidence matrix, default is false", "false");
        System.out.println("输出："+args);
        String [] processedArgs = processArgs(args);
        System.out.println("输出："+processedArgs);
        Map<String, List<String>> parsedArgs = parseArgs(processedArgs);

        final int NUMBER_REDUCER = Integer.parseInt(getOption("numberReducer"));
        final double DECAY_FACTOR = Double.parseDouble(getOption("decayFactor"));
        final int NUM_RANDOM_ITERATION = Integer.parseInt(getOption("numIteration"));
        final int TOP_K_RECOMMENDATION = Integer.parseInt(getOption("topKRecommendation"));
        final String ID_TRANSFORM_PATH = getOption("idTransformPath");
        final String KEY_PREIFX = getOption("keyPrefix");
        final boolean useEvidence = Boolean.parseBoolean(getOption("useEvidence"));
        final boolean useHBase = Boolean.parseBoolean(getOption("useHBase"));


        // first phase is 0.
        AtomicInteger currentPhase = new AtomicInteger();

        /** phase 0,1: convert userId and itemId to Long */
        Path idIndexPath = new Path(tmpBaseOutputPath, ID_DICTIONARY_PATH);
        Path tmpOutputStage1 = new Path(tmpBaseOutputPath, "convert_stage1");
        Path tmpUserNumPath = new Path(tmpBaseOutputPath, "user_num");
        Path tmpItemNumPath = new Path(tmpBaseOutputPath, "item_num");
        Path tmpHBasePrefixPath = new Path(tmpBaseOutputPath, "hbase_prefix");


        Job convertJobStage1 = prepareJob(getInputPath(), tmpOutputStage1,
                TextInputFormat.class,
                IdIndexJob.IdIndexMapper1.class, Text.class, Text.class,
                IdIndexJob.IdIndexReducer1.class, Text.class, NullWritable.class,
                TextOutputFormat.class, NUMBER_REDUCER);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(tmpOutputStage1, true);
            }
            if (!convertJobStage1.waitForCompletion(true)) {
                deletePath(tmpOutputStage1, true);
                return -1;
            }
        }

        Job convertJobStage2 = Job.getInstance(getConf(), "convertJobStage2");
        FileInputFormat.setInputPaths(convertJobStage2, tmpOutputStage1);
        FileOutputFormat.setOutputPath(convertJobStage2, idIndexPath);
        convertJobStage2.setJarByClass(WeightedSimrankRecommendation.class);
        convertJobStage2.setInputFormatClass(TextInputFormat.class);
        convertJobStage2.setMapperClass(IdIndexJob.IdIndexMapper2.class);
        convertJobStage2.setMapOutputKeyClass(Text.class);
        convertJobStage2.setMapOutputValueClass(Text.class);
        convertJobStage2.setReducerClass(IdIndexJob.IdIndexReducer2.class);
        convertJobStage2.setOutputKeyClass(VarLongWritable.class);
        convertJobStage2.setOutputValueClass(Text.class);
        // if do not specify a named ahead, an exception will be thrown, i.e. Named output 'item' not defined
        MultipleOutputs.addNamedOutput(convertJobStage2, IdIndexJob.ITEM, SequenceFileOutputFormat.class, VarLongWritable.class, Text.class);
        MultipleOutputs.addNamedOutput(convertJobStage2, IdIndexJob.USER, SequenceFileOutputFormat.class, VarLongWritable.class, Text.class);
        // prevent generating zero-sized default output, i.e. part-00000
        LazyOutputFormat.setOutputFormatClass(convertJobStage2, SequenceFileOutputFormat.class);
        // use only 1 reducer
        convertJobStage2.setNumReduceTasks(IdIndexJob.getStage2NumberReducer());
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(idIndexPath, true);
            }
            if (!convertJobStage2.waitForCompletion(true)) {
                deletePath(idIndexPath, true);
                return -1;
            }

            int numUser = (int) convertJobStage2.getCounters().findCounter(IdIndexJob.Counters.NUM_USER).getValue();
            int numItem = (int) convertJobStage2.getCounters().findCounter(IdIndexJob.Counters.NUM_ITEM).getValue();
            HadoopUtil.writeInt(numUser, tmpUserNumPath, getConf());
            HadoopUtil.writeInt(numItem, tmpItemNumPath, getConf());

            // this time is use in hbase job.
            final int currentTimestamp = Integer.parseInt(Long.toString(System.currentTimeMillis()).substring(7));
            System.out.println("write hbase prefix:" + currentTimestamp);
            HadoopUtil.writeInt(currentTimestamp, tmpHBasePrefixPath, getConf());

            // store the index in hbase if useHBase is true
            if (useHBase) {

                Path importerHBaseOutput = new Path(tmpBaseOutputPath, "importerHBase");
                Job importerJob = HBaseSupportJob.getIdIndexImporterJob(getConf(), idIndexPath, importerHBaseOutput, NUMBER_REDUCER, currentTimestamp);
                if (deleteOnExist) {
                    deletePath(importerHBaseOutput, true);
                }
                if (!importerJob.waitForCompletion(true)) {
                    deletePath(importerHBaseOutput, true);
                    return -1;
                }
            }
        }


        /**
         * phase 2,3 :
         * append  userId_Long-itemId_Long-rating to userId-itemId-rating data
         * output:
         * two stage, append item and append user.
         */
        Path userItemDataPath = new Path(tmpBaseOutputPath, USER_ITEM_DATA_PATH);
        // append userId
        Path tmpAppendPath1 = new Path(tmpBaseOutputPath, "append_userid");

        Job appendJobStage1 = Job.getInstance(getConf(), "append_userId");
        appendJobStage1.setJarByClass(WeightedSimrankRecommendation.class);
        appendJobStage1.getConfiguration().setInt(IdDataJob.INDEX, IdDataJob.USER_INDEX);
        MultipleInputs.addInputPath(appendJobStage1, getInputPath(), TextInputFormat.class, IdDataJob.IdDataMapper.class);
        MultipleInputs.addInputPath(appendJobStage1, idIndexPath, SequenceFileInputFormat.class, IdDataJob.IdIndexMapper.class);
        FileOutputFormat.setOutputPath(appendJobStage1, tmpAppendPath1);
        appendJobStage1.setReducerClass(IdDataJob.IdDataReducer.class);
        appendJobStage1.setMapOutputValueClass(Text.class);
        appendJobStage1.setMapOutputKeyClass(Text.class);
        appendJobStage1.setOutputKeyClass(Text.class);
        appendJobStage1.setOutputValueClass(NullWritable.class);
        appendJobStage1.setNumReduceTasks(NUMBER_REDUCER);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(tmpAppendPath1, true);
            }
            if (!appendJobStage1.waitForCompletion(true)) {
                deletePath(tmpAppendPath1, true);
                return -1;
            }
        }
        // append itemId
        Job appendJobStage2 = Job.getInstance(getConf(), "append_itemId");
        appendJobStage2.setJarByClass(WeightedSimrankRecommendation.class);
        appendJobStage2.getConfiguration().setInt(IdDataJob.INDEX, IdDataJob.ITEM_INDEX);
        MultipleInputs.addInputPath(appendJobStage2, tmpAppendPath1, TextInputFormat.class, IdDataJob.IdDataMapper.class);
        MultipleInputs.addInputPath(appendJobStage2, idIndexPath, SequenceFileInputFormat.class, IdDataJob.IdIndexMapper.class);
        FileOutputFormat.setOutputPath(appendJobStage2, userItemDataPath);
        appendJobStage2.setReducerClass(IdDataJob.IdDataReducer.class);
        appendJobStage2.setMapOutputValueClass(Text.class);
        appendJobStage2.setMapOutputKeyClass(Text.class);
        appendJobStage2.setOutputKeyClass(Text.class);
        appendJobStage2.setOutputValueClass(NullWritable.class);
        appendJobStage2.setNumReduceTasks(NUMBER_REDUCER);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(userItemDataPath, true);
            }
            if (!appendJobStage2.waitForCompletion(true)) {
                deletePath(userItemDataPath, true);
                return -1;
            }
        }

        /**
         * phase 4: generate weighted transition matrix, Unit matrix and Decay matrix.
         */
        Path matrixPath = new Path(tmpBaseOutputPath, "matrix");
        Path matrixDimensionPath = new Path(tmpBaseOutputPath, "matrix_dimension");
        Path weightTransitionMatrixPath = new Path(matrixPath, "weight_transition_matrix");
        Path unitMatrixPath = new Path(matrixPath, "unit_matrix");
        Path decayMatrixPath = new Path(matrixPath, "decay_matrix");
        Job weigthTransitionMatrixJob = prepareJob(userItemDataPath, weightTransitionMatrixPath,
                TextInputFormat.class,
                WeightedTransitionMatrixJob.WeightedTransitionMatrixMapper.class, VarLongWritable.class, EntityPrefWritable.class,
                WeightedTransitionMatrixJob.WeightedTransitionMatrixReducer.class, IntWritable.class, VectorWritable.class,
                SequenceFileOutputFormat.class, NUMBER_REDUCER);
        weigthTransitionMatrixJob.getConfiguration().set(WeightedTransitionMatrixJob.UNIT_MATRIX_OUTPUT_PATH,
                new Path(unitMatrixPath, "unix").toString());
        weigthTransitionMatrixJob.getConfiguration().set(WeightedTransitionMatrixJob.DECAY_MATRIX_OUTPUT_PATH,
                new Path(decayMatrixPath, "decay").toString());
        weigthTransitionMatrixJob.getConfiguration().setDouble(WeightedTransitionMatrixJob.DECAY_FACTOR, DECAY_FACTOR);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(matrixPath, true);
            }
            if (!weigthTransitionMatrixJob.waitForCompletion(true)) {
                deletePath(matrixPath, true);
                return -1;
            }
            final int MATRIX_DIMENSION = (int) weigthTransitionMatrixJob.getCounters().findCounter(WeightedTransitionMatrixJob.VarianceCounters.DIMENSION).getValue();
            HadoopUtil.writeInt(MATRIX_DIMENSION, matrixDimensionPath, getConf());
        }


        /**
         * phase 5: evidence matrix
         *
         * 2 stages.
         */

        Path evidenceMatrixJobPath = new Path(tmpBaseOutputPath, "evidence");
        // stage 1.
        Path cooccurrencePath = new Path(evidenceMatrixJobPath, "cooccurrence");
        Job evidenceMatrixJob1 = prepareJob(userItemDataPath, cooccurrencePath, TextInputFormat.class,
                EvidenceMatrixJob.CooccurrenceMapper.class, IntWritable.class, IntWritable.class,
                EvidenceMatrixJob.CooccurrenceReducer.class, IntWritable.class, IntWritable.class,
                SequenceFileOutputFormat.class, NUMBER_REDUCER);
        // stage 2.
        Path evidenceJob2Path = new Path(evidenceMatrixJobPath, "stage2");
        Job evidenceMatrixJob2 = prepareJob(cooccurrencePath, evidenceJob2Path, SequenceFileInputFormat.class,
                EvidenceMatrixJob.EvidenceMapper.class, Text.class, IntWritable.class,
                EvidenceMatrixJob.EvidenceReducer.class, Text.class, DoubleWritable.class,
                SequenceFileOutputFormat.class, NUMBER_REDUCER);

        // stage 3.
        Path evidenceMatrixPath = new Path(evidenceMatrixJobPath, "evidence_matrix");
        Job evidenceMatrixJob3 = prepareJob(evidenceJob2Path, evidenceMatrixPath, SequenceFileInputFormat.class,
                EvidenceMatrixJob.EvidenceMatrixMapper.class, IntWritable.class, VectorWritable.class,
                EvidenceMatrixJob.EvidenceMatrixReducer.class, IntWritable.class, VectorWritable.class,
                SequenceFileOutputFormat.class, NUMBER_REDUCER);

        if (shouldRunNextPhase(parsedArgs, currentPhase) && useEvidence) {
            if (deleteOnExist) {
                deletePath(evidenceMatrixJobPath, true);
            }
            if (!evidenceMatrixJob1.waitForCompletion(true)) {
                deletePath(cooccurrencePath, true);
                return -1;
            }

            if (!evidenceMatrixJob2.waitForCompletion(true)) {
                deletePath(evidenceJob2Path, true);
                return -1;
            }
            if (!evidenceMatrixJob3.waitForCompletion(true)) {
                deletePath(evidenceMatrixPath, true);
                return -1;
            }
        }

        final int MATRIX_DIMENSION = HadoopUtil.readInt(matrixDimensionPath, getConf());
        System.out.println("read matrix dimension: " + MATRIX_DIMENSION);

        DistributedRowMatrix evidenceMatrix = new DistributedRowMatrix(evidenceMatrixPath, new Path(evidenceMatrixJobPath, "tmp_matrix"),
                MATRIX_DIMENSION, MATRIX_DIMENSION);
        evidenceMatrix.setConf(getConf());
        /**
         * phase 6: random walk iteration
         */
        Path randomWalkPath = new Path(tmpBaseOutputPath, "random_walk");
        Path randomWalkOutputPath = new Path(randomWalkPath, "output_matrix");
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {

            DistributedRowMatrix similarityMatrix = null;
            if (deleteOnExist) {
                deletePath(randomWalkPath, true);
            }
            try {
                // initially, it is unit matrix.
                similarityMatrix = new DistributedRowMatrix(unitMatrixPath, new Path(randomWalkPath, "tmp_similarity"),
                        MATRIX_DIMENSION, MATRIX_DIMENSION);
                similarityMatrix.setConf(getConf());

                DistributedRowMatrix decayMatrix = new DistributedRowMatrix(decayMatrixPath, new Path(randomWalkPath, "tmp_decay"),
                        MATRIX_DIMENSION, MATRIX_DIMENSION);
                decayMatrix.setConf(getConf());

                DistributedRowMatrix weightedTransitionMatrix = new DistributedRowMatrix(weightTransitionMatrixPath,
                        new Path(randomWalkPath, "tmp_transition"), MATRIX_DIMENSION, MATRIX_DIMENSION);
                weightedTransitionMatrix.setConf(getConf());

                boolean firstIter = true;
                for (int i = 0; i < NUM_RANDOM_ITERATION; i++) {
                    System.out.println("iteration " + i);
                    RandomWalkJob randomWalkJob = new com.rj.recommendation.mr.algorithms.simrank.RandomWalkJob(similarityMatrix, decayMatrix, weightedTransitionMatrix,
                            new Path(randomWalkPath, "tmp_iter" + i), randomWalkOutputPath,
                            firstIter, MATRIX_DIMENSION, NUMBER_REDUCER);
                    similarityMatrix = randomWalkJob.iterateSimilarityMatrix(getConf());
                    firstIter = false;
                }
            } catch (Exception e) {
                e.printStackTrace();
                // deletePath(randomWalkPath, true);
            }

            if (useEvidence) {
                if (similarityMatrix != null && evidenceMatrix != null) {
                    MatrixOperation.matrixMultiply(evidenceMatrix, similarityMatrix, NUMBER_REDUCER, randomWalkOutputPath,
                            new Path(randomWalkPath, "output_matrix_tmp"));
                }
            }

        }

        if (useHBase) {
            // use hbase to do the next steps

            final int NUM_USER = HadoopUtil.readInt(tmpUserNumPath, getConf());
            final int NUM_ITEM = HadoopUtil.readInt(tmpItemNumPath, getConf());
            System.out.println("number user: " + NUM_USER);
            System.out.println("number item: " + NUM_ITEM);

            Path finalOutput = baseOutputPath;

            int hbasePrefix = HadoopUtil.readInt(tmpHBasePrefixPath, getConf());
            System.out.println("hbase prefix: " + hbasePrefix);
            // phase 7:
            Job hbaseJob = HBaseSupportJob.getDecoderJob(getConf(), randomWalkOutputPath, finalOutput, new Path(ID_TRANSFORM_PATH), NUM_ITEM,
                    KEY_PREIFX, TOP_K_RECOMMENDATION, hbasePrefix, NUMBER_REDUCER, idIndexPath);
            if (shouldRunNextPhase(parsedArgs, currentPhase)) {
                if (deleteOnExist) {
                    deletePath(finalOutput, true);
                }
                if (!hbaseJob.waitForCompletion(true)) {
                    deletePath(finalOutput, true);
                    return -1;
                }
            }

            return 0;
        }


        /**
         * phase 7: decode item id, eliminate user, do some sorting and filtering job.
         */
        Path finalOutput = baseOutputPath;
        Path finalTmpOutput = new Path(tmpBaseOutputPath, "final");
        final int NUM_USER = HadoopUtil.readInt(tmpUserNumPath, getConf());
        final int NUM_ITEM = HadoopUtil.readInt(tmpItemNumPath, getConf());
        System.out.println("number user: " + NUM_USER);
        System.out.println("number item: " + NUM_ITEM);

        Path leftOutput = new Path(finalTmpOutput, "left");
        Job filterAndLeftDecodeJob = Job.getInstance(getConf(), "filter_and_left_decode_job");
        filterAndLeftDecodeJob.getConfiguration().setInt(FinalProcessJob.NUM_ITEM, NUM_ITEM);
        filterAndLeftDecodeJob.setJarByClass(WeightedTransitionMatrixJob.class);
        filterAndLeftDecodeJob.getConfiguration().setInt(FinalProcessJob.INDEX, FinalProcessJob.LEFT);
        MultipleInputs.addInputPath(filterAndLeftDecodeJob, randomWalkOutputPath,
                SequenceFileInputFormat.class, FinalProcessJob.SimilarityMatrixFilterMapper.class);
        MultipleInputs.addInputPath(filterAndLeftDecodeJob, idIndexPath,
                SequenceFileInputFormat.class, FinalProcessJob.IdIndexReaderMapper.class);
        filterAndLeftDecodeJob.setOutputFormatClass(TextOutputFormat.class);
        FileOutputFormat.setOutputPath(filterAndLeftDecodeJob, leftOutput);
        filterAndLeftDecodeJob.setReducerClass(FinalProcessJob.JoinReducer.class);
        filterAndLeftDecodeJob.setMapOutputKeyClass(IntWritable.class);
        filterAndLeftDecodeJob.setMapOutputValueClass(Text.class);
        filterAndLeftDecodeJob.setOutputKeyClass(NullWritable.class);
        filterAndLeftDecodeJob.setOutputValueClass(Text.class);
        filterAndLeftDecodeJob.setNumReduceTasks(NUMBER_REDUCER);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(leftOutput, true);
            }

            if (!filterAndLeftDecodeJob.waitForCompletion(true)) {
                deletePath(leftOutput, true);
                return -1;
            }
        }

        /**
         * phase 8: decode item id, eliminate user, do some sorting and filtering job.
         */
        Path rightOutput = new Path(finalTmpOutput, "right");
        Job rightDecodeJob = Job.getInstance(getConf(), "right_decode_job");
        rightDecodeJob.getConfiguration().setInt(FinalProcessJob.INDEX, FinalProcessJob.RIGHT);
        rightDecodeJob.setJarByClass(WeightedTransitionMatrixJob.class);
        MultipleInputs.addInputPath(rightDecodeJob, leftOutput, TextInputFormat.class,
                FinalProcessJob.SimilarityReaderMapper.class);
        MultipleInputs.addInputPath(rightDecodeJob, idIndexPath, SequenceFileInputFormat.class,
                FinalProcessJob.IdIndexReaderMapper.class);
        rightDecodeJob.setOutputFormatClass(TextOutputFormat.class);
        FileOutputFormat.setOutputPath(rightDecodeJob, rightOutput);
        rightDecodeJob.setReducerClass(FinalProcessJob.JoinReducer.class);
        rightDecodeJob.setMapOutputKeyClass(IntWritable.class);
        rightDecodeJob.setMapOutputValueClass(Text.class);
        rightDecodeJob.setOutputKeyClass(NullWritable.class);
        rightDecodeJob.setOutputValueClass(Text.class);
        rightDecodeJob.setNumReduceTasks(NUMBER_REDUCER);
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(rightOutput, true);
            }

            if (!rightDecodeJob.waitForCompletion(true)) {
                deletePath(rightOutput, true);
                return -1;
            }
        }

        /**
         * phase 9: item stringId truncation, non-ad filtering, ad sorting..
         */
        Path cachedIdTransformPath = new Path(ID_TRANSFORM_PATH);
        Job cachedIdTransformJob = prepareJob(rightOutput, finalOutput, TextInputFormat.class, FinalProcessJob.FilterAndSortMapper.class, Text.class, Text.class,
                FinalProcessJob.FilterAndSortReducer.class, Text.class, NullWritable.class, TextOutputFormat.class, NUMBER_REDUCER);
        cachedIdTransformJob.getConfiguration().set(FinalProcessJob.KEY_PREFIX, KEY_PREIFX);
        cachedIdTransformJob.getConfiguration().setInt(FinalProcessJob.TOP_K, TOP_K_RECOMMENDATION);
        if (fs.exists(cachedIdTransformPath)) {
            cachedIdTransformJob.addCacheFile(new URI(cachedIdTransformPath.toUri() + "#" + FinalProcessJob.CACHE_FILE));
        } else {
            throw new Exception("there is no cache file in " + cachedIdTransformPath.toString());
        }
        if (shouldRunNextPhase(parsedArgs, currentPhase)) {
            if (deleteOnExist) {
                deletePath(finalOutput, true);
            }

            if (!cachedIdTransformJob.waitForCompletion(true)) {
                deletePath(finalOutput, true);
                return -1;
            }
        }


        return 0;

    }

}
