package uk.ac.shef.inf.msm4phi.indexing;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import twitter4j.ResponseList;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import uk.ac.shef.inf.msm4phi.Util;

import java.util.*;

/**
 * Process the collected tweets in the solr index to add information that is not available during streaming:
 * <p>
 * This is a single thread process to avoid updating a solr index by multiple processes at the same time.
 * <p>
 * <p>
 * <br/> favorite_count
 * <br/> retweet_count
 * <br/> retweeted - always false. use 'retweet_count>0' for this purpose
 * <br/> quote_count - not available from twit4j
 */
public class PostProcessor {
    private static final Logger LOG = Logger.getLogger(PostProcessor.class.getName());

    private SolrClient solrClient;
    private final int MAX_LOOKUP_ID_PER_CALL = 100;
    private final int SEC_BETWEEN_CALLS = 15 * 60 / 900;  //900 calls per 15 minutes.
    private final int BATCH_SIZE = 9000;
    private Twitter twitter;
    private int twitterCalls=0;

    public PostProcessor(SolrClient solrClient,
                         String twitterConsumerKey, String twitterConsumerSecret,
                         String twitterAccessToken, String twitterAccessSecret) {
        this.solrClient = solrClient;
        this.twitter = Util.authenticateTwitter(twitterConsumerKey, twitterConsumerSecret, twitterAccessToken,
                twitterAccessSecret);
    }

    public void process() {
        SolrQuery q = new SolrQuery();
        q.setQuery("*:*");
        q.setStart(0);
        q.setRows(BATCH_SIZE);
        q.setFields("*");

        boolean stop = false;
        while (!stop) {
            QueryResponse res = null;
            long total = 0;
            try {
                res = Util.performQuery(q, solrClient);
                if (res != null) {
                    total = res.getResults().getNumFound();
                    //update results
                    LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                            total, q.getStart(), q.getStart() + q.getRows()));
                    updateBatch(res, solrClient);
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


    }

    /**
     * <br/> favorite_count
     * <br/> retweet_count
     * <br/> retweeted
     *
     * @param res
     */
    private void updateBatch(QueryResponse res, SolrClient solrClient) {
        Map<String, SolrDocument> lookupIDs = new HashMap<>();
        twitterCalls=0;
        int count=0;
        for (SolrDocument d : res.getResults()) {
            count++;
            String tweetID = d.get("id").toString();
            lookupIDs.put(tweetID, d);
            List<SolrInputDocument> updates = new ArrayList<>();

            if (lookupIDs.size() == MAX_LOOKUP_ID_PER_CALL || count==res.getResults().size()) {
                //make call
                try {
                    ResponseList<Status> tweets = makeTwitterLookupCall(lookupIDs.keySet());

                    //update each solr doc based on the twitter api results
                    for (Status tw : tweets) {
                        if (tw.getFavoriteCount() > 0 || tw.getRetweetCount() > 0) {
                            SolrInputDocument updateDoc = new SolrInputDocument();
                            updateDoc.addField("id", tw.getId());
                            if (tw.getFavoriteCount() > 0) {
                                Map<String, Object> fieldFavModifier = new HashMap<>(1);
                                fieldFavModifier.put("set", tw.getFavoriteCount());
                                updateDoc.addField("favorite_count", fieldFavModifier);
                            }// add the map as the field value
                            if (tw.getRetweetCount() > 0) {
                                Map<String, Object> fieldRTModifier = new HashMap<>(1);
                                fieldRTModifier.put("set", tw.getRetweetCount());
                                updateDoc.addField("retweet_count", fieldRTModifier);

                            }// add the map as the field value
                            updates.add(updateDoc);
                        }
                    }

                } catch (TwitterException e) {
                    LOG.warn(String.format("\t\tunable to query Twitter, ids are: %s \n\t %s",
                            lookupIDs.keySet().toString(), ExceptionUtils.getFullStackTrace(e)));
                }

                //reset container
                lookupIDs.clear();

                try {
                    solrClient.add(updates);
                    solrClient.commit();
                    Thread.sleep(SEC_BETWEEN_CALLS * 1000);
                } catch (Exception e) {
                    LOG.warn(String.format("\t\tfailed to commit to solr with exception: \n\t %s",
                            ExceptionUtils.getFullStackTrace(e)));
                }
            }
        }
    }


    private ResponseList<Status> makeTwitterLookupCall(Set<String> strings) throws TwitterException {
        twitterCalls++;
        LOG.info(String.format("\t\t\tmaking %dth Twitter call for a batch of max %d Tweets, total ids to query =%d",
                twitterCalls,BATCH_SIZE,strings.size()));
        long[] ids = new long[strings.size()];
        String[] idStr = strings.toArray(new String[0]);
        for (int i = 0; i < ids.length; i++)
            ids[i] = Long.valueOf(idStr[i]);
        return twitter.lookup(ids);
    }


}
