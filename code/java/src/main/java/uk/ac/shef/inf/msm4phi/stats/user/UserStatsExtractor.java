package uk.ac.shef.inf.msm4phi.stats.user;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.IndexAnalyserWorker;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.util.*;

public class UserStatsExtractor extends IndexAnalyserWorker {
    private static final Logger LOG = Logger.getLogger(UserStatsExtractor.class.getName());
    private StandardDeviation sd = new StandardDeviation();
    private final int MIN_INSTNACES_FOR_OUTLIER=10;

    public UserStatsExtractor(int id, SolrClient solrClient, String outFolder) {
        super(id, solrClient, outFolder);
    }

    @Override
    protected int computeSingleWorker(Map<String, List<String>> tasks) {
        CSVWriter csvWriter = null;
        int totalHashtags=0;
        try {
            csvWriter = Util.createCSVWriter(outFolder + "/" + id + ".csv");
            writeCSVHeader(csvWriter);

            for (Map.Entry<String, List<String>> en : tasks.entrySet()) {
                totalHashtags++;
                /*if (en.getValue().contains("heartdisease"))
                    System.out.println();*/
                LOG.info(String.format("\t processing hashtag '%s' with %d variants...", en.getKey(), en.getValue().size()));
                SolrQuery q = Util.createQueryUsersOfHashtags(resultBatchSize, en.getValue().toArray(new String[0]));

                List<Double> arrayUserNewTweets = new ArrayList<>();
                List<Double> arrayUserReTweets = new ArrayList<>();
                List<Double> arrayCGFollowers = new ArrayList<>();
                List<Double> arrayCPFollowers = new ArrayList<>();
                long users = 0, total_nt=0, total_rt=0, cg = 0, cp = 0, cg_follower = 0, cp_follower = 0;

                boolean stop = false;
                while (!stop) {
                    QueryResponse res = null;
                    try {
                        res = Util.performQuery(q, solrClient);
                        if (res != null)
                            users = res.getResults().getNumFound();
                        //update results
                        LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                                users, q.getStart(), q.getStart() + q.getRows()));
                        for (SolrDocument d : res.getResults()) {
                            //followers
                            int followers=getCount(d, "user_followers_count");

                            //_content_gen
                            int nt =getCount(d, "user_newtweet_count");
                            int reply = getCount(d, "user_reply_count");
                            nt+=reply;
                            if (nt>0) {
                                total_nt+=nt;
                                cg++;
                                arrayUserNewTweets.add((double)nt);
                                cg_follower+=followers;
                                arrayCGFollowers.add((double)followers);
                            }

                            //_content_prop
                            int rt =getCount(d, "user_retweet_count");
                            int quote = getCount(d, "user_quote_count");
                            rt+=quote;
                            if (rt>0) {
                                total_rt+=rt;
                                cp++;
                                arrayUserReTweets.add((double)nt);
                                cp_follower+=followers;
                                arrayCPFollowers.add((double)followers);
                            }
                        }

                    } catch (Exception e) {
                        LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                                q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
                    }

                    int curr = q.getStart() + q.getRows();
                    if (curr < users)
                        q.setStart(curr);
                    else
                        stop = true;

                }
                //calculating deviations and outliers
                LOG.info(String.format("\t calculating SD and outliers for CG (elements of %d)...",
                        arrayUserNewTweets.size()));
                double avg_nt_per_cg = cg==0?0:(double) total_nt / cg;
                double[] primitives=ArrayUtils.toPrimitive(arrayUserNewTweets.toArray(new Double[0]));
                double dev_nt=primitives.length>0?sd.evaluate(primitives):0;
                List<List<Double>> outliers=primitives.length<MIN_INSTNACES_FOR_OUTLIER
                        ?null:Util.detectOutliersIQR(primitives);
                double[] nt_outlier_stats= calculateOutlierStats(outliers, primitives.length);

                LOG.info(String.format("\t calculating SD and outliers for CP (elements of %d)...",
                        arrayUserReTweets.size()));
                double avg_rt_per_cp=cp==0?0: (double)total_rt / cp;
                primitives=ArrayUtils.toPrimitive(arrayUserReTweets.toArray(new Double[0]));
                double dev_rt=primitives.length>0?sd.evaluate(primitives):0;
                outliers=primitives.length<MIN_INSTNACES_FOR_OUTLIER
                        ?null:Util.detectOutliersIQR(primitives);
                double[] rt_outlier_stats= calculateOutlierStats(outliers, primitives.length);

                LOG.info(String.format("\t calculating SD and outliers for CG followers (elements of %d)...",
                        arrayCGFollowers.size()));
                double cg_reach = cg==0?0: (double)cg_follower / cg;
                primitives=ArrayUtils.toPrimitive(arrayCGFollowers.toArray(new Double[0]));
                double dev_cgr=primitives.length>0?sd.evaluate(primitives):0;
                outliers=primitives.length<MIN_INSTNACES_FOR_OUTLIER
                        ?null:Util.detectOutliersIQR(primitives);
                double[] cgr_outlier_stats= calculateOutlierStats(outliers, primitives.length);

                LOG.info(String.format("\t calculating SD and outliers for CP followers (elements of %d)...",
                        arrayCPFollowers.size()));
                double cp_reach = cp==0?0: (double)cp_follower / cp;
                primitives=ArrayUtils.toPrimitive(arrayCPFollowers.toArray(new Double[0]));
                double dev_cpr=primitives.length>0?sd.evaluate(primitives):0;
                outliers=primitives.length<MIN_INSTNACES_FOR_OUTLIER
                        ?null:Util.detectOutliersIQR(primitives);
                double[] cpr_outlier_stats= calculateOutlierStats(outliers, primitives.length);

                //prepare line to write to csv
                String[] line = new String[20];
                line[0] = en.getKey();
                line[1] = String.valueOf(users);
                line[2] = String.format("%.4f", users==0?0:(double) cg / users);
                line[3] = String.format("%.4f", users==0?0:(double) cp / users);
                line[4] = String.format("%.4f", avg_nt_per_cg);
                line[5] = String.format("%.4f", dev_nt);
                line[6] = String.format("%.4f", nt_outlier_stats[0]);
                line[7] = String.format("%.4f", nt_outlier_stats[1]);

                line[8] = String.format("%.4f", (double) avg_rt_per_cp);
                line[9] = String.format("%.4f", (double) dev_rt);
                line[10] = String.format("%.4f", rt_outlier_stats[0]);
                line[11] = String.format("%.4f", rt_outlier_stats[1]);

                line[12] = String.format("%.4f", (double) cg_reach);
                line[13] = String.format("%.4f", dev_cgr);
                line[14] = String.format("%.4f", cgr_outlier_stats[0]);
                line[15] = String.format("%.4f", cgr_outlier_stats[1]);

                line[16] = String.format("%.4f", (double) cp_reach);
                line[17] = String.format("%.4f", dev_cpr);
                line[18] = String.format("%.4f", cpr_outlier_stats[0]);
                line[19] = String.format("%.4f", cpr_outlier_stats[1]);
                csvWriter.writeNext(line);
            }

            csvWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return totalHashtags;
    }

    @Override
    protected IndexAnalyserWorker createInstance(Map<String, List<String>> splitTasks, int id) {
        UserStatsExtractor extractor = new UserStatsExtractor(
                id, solrClient,outFolder
        );
        extractor.setHashtagMap(splitTasks);
        extractor.setMaxTasksPerThread(maxTasksPerThread);
        return extractor;
    }

    /**
     * unique_users: unique user id associated with a tag
     * %_content_gen: % of unique users that have created new content
     * %_content_pp: % of unique users that have propagated content, e.g., retweet, quote
     * avg_nt_per_cg: average # of new tweets per content generator
     * dev_nt: standard deviation of new tweets generated by all content generators
     * %cg_outliers+: # of users that are considered outliers given their new tweet authoring behaviour, above mean
     * %cg_outliers-: # of users that are considered outliers given their new tweet authoring behaviour, below mean
     * avg_rt_per_cg: average # of retweets per content propagator
     * dev_rt: standard deviation of retweets generated by all content propagators
     * %cp_outliers+: # of users that are considered outliers given their retweet behaviour, above mean
     * %cp_outliers-: # of users that are considered outliers given their retweet behaviour, below mean
     * cg_reach: average # of followers per content generator
     * dev_cgr: standard deviation of followers of content generators
     * %cgr_outliers+: # of content generators whose # of followers are considered to be outlier, above mean
     * %cgr_outliers-: # of content generators whose # of followers are considered to be outlier, below mean
     * cp_reach: average # of followers per content propagator
     * dev cpr: standard deviation of followers of content propagators
     * %cpr_outliers+: # of content propagators whose # of followers are considered to be outlier, above mean
     * %cpr_outliers-: # of content propagators whose # of followers are considered to be outlier, below mean
     *
     * @param csvWriter
     */
    private void writeCSVHeader(CSVWriter csvWriter) {
        //0         1       2
        String[] headerRecord = {"tag", "unique_users", "%_content_gen",
                //3             4               5              6               7
                "%_content_pp", "avg_nt_per_cg", "dev_nt", "%cg_outliers-", "%cg_outliers+",
                //8                9                 10         11
                "avg_rt_per_cp", "dev_rt", "%cp_outliers-","%cp_outliers+",
                //12          13               14          15
                "cg_reach", "dev_cgr", "cgr_outliers-","cgr_outliers+",
                //16        17          18              19
                "cp_reach", "dev_cpr", "cpr_outliers-","cpr_outliers+"};
        csvWriter.writeNext(headerRecord);
    }

    private int getCount(SolrDocument d, String field){
        Object nt=d.getFieldValue(field);
        if (nt!=null){
            int nt_=Integer.valueOf(nt.toString());
            return nt_;
        }
        return 0;
    }

    private double[] calculateOutlierStats(List<List<Double>> detectedValues, int arraySize){
        if (detectedValues==null)
            return new double[]{-1,-1};

        return new double[]{(double)detectedValues.get(0).size()/arraySize, (
                double)detectedValues.get(1).size()/arraySize};
    }
}
