package uk.ac.shef.inf.msm4phi.stats.interaction;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.IndexAnalyserWorker;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.util.*;

public class InteractionStatsExtractor extends IndexAnalyserWorker {
    private static final Logger LOG = Logger.getLogger(InteractionStatsExtractor.class.getName());

    public InteractionStatsExtractor(int id, SolrClient solrClient, String outFolder) {
        super(id, solrClient, outFolder);
    }

    @Override
    protected int computeSingleWorker(Map<String, List<String>> tasks) {
        CSVWriter csvWriter = null;
        int totalHashtags = 0;
        try {
            csvWriter = Util.createCSVWriter(outFolder + "/" + id + ".csv");
            writeCSVHeader(csvWriter);

            for (Map.Entry<String, List<String>> en : tasks.entrySet()) {
                totalHashtags++;
                LOG.info(String.format("\t processing hashtag '%s' with %d variants...", en.getKey(), en.getValue().size()));
                SolrQuery q = Util.createQTOHWithInteractions(resultBatchSize, en.getValue().toArray(new String[0]));
                SolrQuery qGeneral = Util.createQueryTweetsOfHashtags(10, en.getValue().toArray(new String[0]));
                long retweeted = 0, rt_freq = 0, quoted = 0, quote_freq = 0,
                        liked = 0, like_freq = 0, replied = 0, reply_freq = 0, as_replies = 0, as_quotes = 0,
                        interacted = 0;
                ;

                long total = 0, totalGeneral = 0;
                boolean stop = false;
                while (!stop) {
                    QueryResponse res = null, resGeneral = null;
                    try {
                        res = Util.performQuery(q, solrClient);
                        resGeneral = Util.performQuery(qGeneral, solrClient);
                        if (res != null && resGeneral != null) {
                            total = res.getResults().getNumFound();
                            totalGeneral = res.getResults().getNumFound();
                        }
                        //update results
                        LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                                total, q.getStart(), q.getStart() + q.getRows()));

                        boolean hasInteraction = false;
                        for (SolrDocument d : res.getResults()) {
                            Object rt = d.getFieldValue("retweet_count");
                            if (rt != null) {
                                int rt_ = Integer.valueOf(rt.toString());
                                if (rt_ > 0) {
                                    retweeted++;
                                    rt_freq += rt_;
                                    if (!hasInteraction) {
                                        interacted++;
                                        hasInteraction = true;
                                    }
                                }
                            }
                            Object qt = d.getFieldValue("quote_count");
                            if (qt != null) {
                                int qt_ = Integer.valueOf(qt.toString());
                                if (qt_ > 0) {
                                    quoted++;
                                    quote_freq += qt_;
                                    if (!hasInteraction) {
                                        interacted++;
                                        hasInteraction = true;
                                    }
                                }
                            }
                            Object fav = d.getFieldValue("favorite_count");
                            if (fav != null) {
                                int fav_ = Integer.valueOf(fav.toString());
                                if (fav_ > 0) {
                                    liked++;
                                    like_freq += fav_;
                                    if (!hasInteraction) {
                                        interacted++;
                                        hasInteraction = true;
                                    }
                                }
                            }
                            Object rp = d.getFieldValue("reply_count");
                            if (rp != null) {
                                int rp_ = Integer.valueOf(rp.toString());
                                if (rp_ > 0) {
                                    replied++;
                                    reply_freq += rp_;
                                    if (!hasInteraction)
                                        interacted++;
                                }
                            }

                            Object ar = d.getFieldValue("in_reply_to_status_id_str");
                            if (ar != null && !ar.toString().equalsIgnoreCase(""))
                                as_replies++;

                            Object aq = d.getFieldValue("quoted_status_id_str");
                            if (aq != null&& !aq.toString().equalsIgnoreCase(""))
                                as_quotes++;

                            hasInteraction = false;
                        }

                    } catch (Exception e) {
                        LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                                q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
                    }

                    int curr = q.getStart() + q.getRows();
                    if (curr < total)
                        q.setStart(curr);
                    else
                        stop = true;

                }
                //prepare line to write to csv
                String[] line = new String[12];
                line[0] = en.getKey();
                line[1] = String.valueOf(total);
                if (totalGeneral == 0)
                    for (int i = 2; i < 10; i++)
                        line[i] = "0";
                else {
                    line[2] = String.format("%.4f", (double) (retweeted + quoted) / totalGeneral);
                    line[3] = String.format("%.4f", (double) (rt_freq + quote_freq) / totalGeneral);
                    line[4] = String.format("%.4f", (double) liked / totalGeneral);
                    line[5] = String.format("%.4f", (double) like_freq / totalGeneral);
                    line[6] = String.format("%.4f", (double) replied / totalGeneral);
                    line[7] = String.format("%.4f", (double) reply_freq / totalGeneral);
                    line[8] = String.format("%.4f", (double) interacted / totalGeneral);
                    line[9] = String.format("%.4f",
                            (double) (rt_freq + quote_freq + like_freq + reply_freq) / totalGeneral);
                    line[10] = String.format("%.4f",
                            (double) (as_replies) / totalGeneral);
                    line[11] = String.format("%.4f",
                            (double) (as_quotes) / totalGeneral);
                }

                csvWriter.writeNext(line);
            }

            csvWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return totalHashtags;
    }

    /**
     * Schema:
     *
     * @param csvWriter
     */
    private void writeCSVHeader(CSVWriter csvWriter) {
        //0         1       2
        String[] headerRecord = {"tag", "total", "%_retweeted",
                //3             4               5              6
                "avg_rt_freq", "%_liked", "avg_like_freq", "%_replied",
                //7                     8                      9
                "avg_reply_freq", "%_interacted", "avg_interact_freq","as_replies","as_quotes"};
        csvWriter.writeNext(headerRecord);
    }

    @Override
    protected InteractionStatsExtractor createInstance(Map<String, List<String>> splitTasks, int id) {
        InteractionStatsExtractor extractor = new InteractionStatsExtractor(
                id, solrClient, outFolder
        );
        extractor.setHashtagMap(splitTasks);
        extractor.setMaxTasksPerThread(maxTasksPerThread);
        return extractor;
    }
}
