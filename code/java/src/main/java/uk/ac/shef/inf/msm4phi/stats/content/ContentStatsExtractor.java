package uk.ac.shef.inf.msm4phi.stats.content;

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

public class ContentStatsExtractor extends IndexAnalyserWorker {
    private static final Logger LOG = Logger.getLogger(ContentStatsExtractor.class.getName());

    public ContentStatsExtractor(int id, SolrClient solrClient, String outFolder) {
        super(id, solrClient, outFolder);
    }

    @Override
    protected int computeSingleWorker(Map<String, List<String>> tasks) {
        int totalHashtags=0;
        CSVWriter csvWriter = null;
        try {
            csvWriter = Util.createCSVWriter(outFolder + "/" + id + ".csv");
            writeCSVHeader(csvWriter);

            for (Map.Entry<String, List<String>> en : tasks.entrySet()) {
                /*if (en.getValue().contains("hepb"))
                    System.out.println();*/
                totalHashtags++;
                LOG.info(String.format("\t processing hashtag '%s' with %d variants...", en.getKey(), en.getValue().size()));
                SolrQuery q = Util.createQueryTweetsOfHashtags(resultBatchSize, en.getValue().toArray(new String[0]));

                Set<String> uniqueHashtags = new HashSet<>();
                Set<String> uniqueMentions = new HashSet<>();
                Set<String> uniqueURLs = new HashSet<>();
                Set<String> uniqueMedia = new HashSet<>();
                Set<String> uniqueSymbol = new HashSet<>();
                Set<String> uniqueWords = new HashSet<>();
                long hashtags = 0, mentions = 0, urls = 0, media = 0, symbols = 0,
                        length = 0;
                long total = 0;

                boolean stop = false;
                while (!stop) {
                    QueryResponse res = null;
                    try {
                        res = Util.performQuery(q, solrClient);
                        if (res != null)
                            total = res.getResults().getNumFound();
                        //update results
                        LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                                total, q.getStart(), q.getStart() + q.getRows()));
                        for (SolrDocument d : res.getResults()) {
                            String[] words = d.getFieldValue("status_text").toString().toLowerCase().split("\\s+");
                            uniqueWords.addAll(Arrays.asList(words));
                            length += words.length;

                            Set<String> docURLs = getSetObjects(d, "entities_url");
                            urls += docURLs.size();
                            uniqueURLs.addAll(docURLs);
                            Set<String> docHashtags = getSetObjects(d, "entities_hashtag");
                            hashtags += docHashtags.size();
                            uniqueHashtags.addAll(docHashtags);
                            Set<String> docMedia = getSetObjects(d, "entities_media_url");
                            media += docMedia.size();
                            uniqueMedia.addAll(docMedia);
                            Set<String> docMentions = getSetObjects(d, "entities_user_mention");
                            mentions += docMentions.size();
                            uniqueMentions.addAll(docMentions);
                            Set<String> docSymbols = getSetObjects(d, "entities_symbol");
                            symbols += docSymbols.size();
                            uniqueSymbol.addAll(docSymbols);
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
                String[] line = new String[17];
                line[0] = en.getKey();
                line[1] = String.valueOf(total);
                if (total==0)
                    for (int i=2; i<17; i++)
                        line[i]="0";
                else {
                    line[2] = String.format("%.4f", (double) uniqueHashtags.size() / total);
                    line[3] = String.format("%.4f", (double) uniqueMentions.size() / total);
                    line[4] = String.format("%.4f", (double) uniqueURLs.size() / total);
                    line[5] = String.format("%.4f", (double) uniqueMedia.size() / total);
                    line[6] = String.format("%.4f", (double) uniqueSymbol.size() / total);
                    uniqueHashtags.addAll(uniqueMentions);
                    uniqueHashtags.addAll(uniqueURLs);
                    uniqueHashtags.addAll(uniqueMedia);
                    uniqueHashtags.addAll(uniqueSymbol);
                    line[7] = String.format("%.4f", (double) uniqueHashtags.size() / total);

                    line[8] = String.format("%.4f", (double) hashtags / total);
                    line[9] = String.format("%.4f", (double) mentions / total);
                    line[10] = String.format("%.4f", (double) urls / total);
                    line[11] = String.format("%.4f", (double) media / total);
                    line[12] = String.format("%.4f", (double) symbols / total);
                    line[13] = String.format("%.4f", (double) (hashtags + mentions + urls + media + symbols) / total);
                    line[14] = String.format("%.4f", (double) uniqueWords.size() / total);
                    line[15] = String.format("%.4f", (double) length / total);
                    line[16] = String.format("%.4f", (double) uniqueWords.size() / length);
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
     * <br/> total_tweets
     * <br/> avg_unique_hashtag -> average unique hashtags per tweet
     * <br/> avg_unique_mention -> average unique mentions per tweet
     * <br/> avg_unique_urls -> average unique urls per tweet
     * <br/> avg_unique_media -> average unique media per tweet
     * <br/> avg_unique_symbol-> average unique symbols
     * <br/> avg_unique_anyentity -> average unique entity (any kind) per tweet
     * <br/> avg_hashtags -> average hashtags per tweet (# of hashtags in each tweet are added up then divided)
     * <br/> avg_mention -> average mentions per tweet
     * <br/> avg_urls -> average urls per tweet
     * <br/> avg_media -> average media per tweet
     * <br/> avg_symbol-> average symbols
     * <br/> avg_anyentity -> average entity (any kind) per tweet
     * <br/> avg_tweet_uniquewords  -> unique words as set, divided by # of tweets
     * <br/> avg_tweet_length -> lengths by words
     * <br/> avg_tweet_density -> uniquewords/total length
     *
     * @param csvWriter
     */
    private void writeCSVHeader(CSVWriter csvWriter) {
        //0         1
        String[] headerRecord = {"tag", "total_tweets",
                //2                     3                       4                     5
                "avg_unique_hashtag", "avg_unique_mention", "avg_unique_urls", "avg_unique_media",
                //6                     7                         8                  9            10
                "avg_unique_symbol", "avg_unique_anyentity", "avg_hashtags", "avg_mention", "avg_urls",
                //11            12              13                      14          15
                "avg_media", "avg_symbol", "avg_anyentity", "avg_tweet_uniquewords","avg_tweet_length", "avg_tweet_density"};
        csvWriter.writeNext(headerRecord);
    }

    private Set<String> getSetObjects(SolrDocument d, String field) {
        Set<String> set = new HashSet<>();
        Collection<Object> values = d.getFieldValues(field);
        if (values != null) {
            for (Object o : values) {
                set.add(o.toString());
            }
        }
        return set;
    }


    @Override
    protected ContentStatsExtractor createInstance(Map<String, List<String>> splitTasks, int id) {
        ContentStatsExtractor extractor = new ContentStatsExtractor(id, this.solrClient, outFolder);
        extractor.setHashtagMap(splitTasks);
        extractor.setMaxTasksPerThread(this.maxTasksPerThread);
        return extractor;
    }
}
