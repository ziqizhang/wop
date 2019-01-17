package uk.ac.shef.inf.msm4phi.indexing;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import twitter4j.*;
import uk.ac.shef.inf.msm4phi.Util;

import java.util.*;

public class IndexPopulatorUser {
    private static final Logger LOG = Logger.getLogger(IndexPopulatorUser.class.getName());

    private SolrClient tweetCoreClient;
    private SolrClient userCoreClient;
    private Twitter twitter;
    private Date prevCallDate;

    public IndexPopulatorUser(SolrClient tweetCore, SolrClient userCore,
                              String twitterConsumerKey, String twitterConsumerSecret,
                              String twitterAccessToken, String twitterAccessSecret) {
        this.tweetCoreClient = tweetCore;
        this.userCoreClient = userCore;
        this.twitter = Util.authenticateTwitter(twitterConsumerKey, twitterConsumerSecret, twitterAccessToken,
                twitterAccessSecret);
    }

    public void process() {
        LOG.info(String.format("Getting unique users from the Tweet core... %s", new Date().toString()));
        List<String> uniqueUsers = new ArrayList<>(getUniqueUsersFromSolr(tweetCoreClient));
        LOG.info(String.format("Total of %d unique users to process", uniqueUsers.size()));

        prevCallDate = new Date();
        try {
            Thread.sleep(1100);
        } catch (InterruptedException e) {
        }


        int countU = 0;
        Iterator<String> userIt = uniqueUsers.iterator();
        Map<String, User> preFetchedUsers = new LinkedHashMap<>();
        while (userIt.hasNext()) {
            String userID = userIt.next();

            if (!preFetchedUsers.containsKey(userID))
                preFetchedUsers = 100 < uniqueUsers.size() ? makeTwitterLookupCall(uniqueUsers.subList(0, 100)) :
                        makeTwitterLookupCall(uniqueUsers.subList(0, uniqueUsers.size()));

            LOG.info(String.format("\t currently processing #%d, %s", countU, userID));
            //Twitter API part
            SolrInputDocument newDoc = new SolrInputDocument();
            newDoc.addField("id", userID);
            populateFromTwitterAPI(userID, newDoc, preFetchedUsers);

            //solr index part
            populateFromTweetIndex(userID, newDoc, tweetCoreClient);

            //commit
            userIt.remove();
            countU++;
            try {
                userCoreClient.add(newDoc);
            } catch (Exception e) {
                LOG.warn(String.format("\t\tfailed to add user data to index: %s \n\t %s",
                        userID, ExceptionUtils.getFullStackTrace(e)));
            }
            if (countU % 500 == 0) {
                try {
                    userCoreClient.commit();
                } catch (Exception e) {
                    LOG.warn(String.format("\t\tfailed to commit to solr with exception: \n\t %s",
                            ExceptionUtils.getFullStackTrace(e)));
                }
            }
        }


        //commit
        try {
            userCoreClient.commit();
        } catch (Exception e) {
            LOG.warn(String.format("\t\tfailed to commit to solr with exception: \n\t %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    /**
     * fetch data from the Tweet index to populate user data
     * <p>
     * <!--the new tweets, retweets, replies, quotes, favorites by this user, based on the data
     * collected in the 'tweets' index-->
     * <field name="user_newtweet_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_retweet_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_reply_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_quote_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <p>
     * <!--frequency the user's tweet/retweet is liked, retweeted, quoted, or replied to-->
     * <field name="user_favorited_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_retweeted_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_quoted_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_replied_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <p>
     * <field name="user_entities_hashtag" type="string" indexed="true" stored="true" multiValued="true"/>
     * <field name="user_entities_symbol" type="string" indexed="true" stored="true" multiValued="true"/>
     * <field name="user_entities_url" type="string" indexed="true" stored="true" multiValued="true"/>
     * <field name="user_entities_user_mention" type="string" indexed="true" stored="true" multiValued="true"/>
     * <field name="user_entities_media_url" type="string" indexed="true" stored="true" multiValued="true"/>
     * <field name="user_entities_media_type" type="string" indexed="true" stored="true" multiValued="true"/>
     *
     * @param userID
     * @param newDoc
     */
    private void populateFromTweetIndex(String userID, SolrInputDocument newDoc, SolrClient solrClient) {
        SolrQuery q = Util.createQueryTweetsOfUser(10000, userID);
        boolean stop = false;

        int newTweet = 0, retweet = 0, reply = 0, quote = 0,
                favorited = 0, retweeted = 0, quoted = 0, replied = 0;
        Set<String> hashtags = new HashSet<>(),
                symbols = new HashSet<>(),
                urls = new HashSet<>(),
                mentions = new HashSet<>(),
                mediaURLs = new HashSet<>(),
                mediaTypes = new HashSet<>();

        while (!stop) {
            QueryResponse res = null;
            long total = 0;
            try {
                res = Util.performQuery(q, solrClient);
                if (res != null) {
                    total = res.getResults().getNumFound();
                    //update results
                    LOG.info(String.format("\t\ttotal tweets of user %s is %d, currently processing from %d to %d...",
                            userID, total, q.getStart(), q.getStart() + q.getRows()));
                    for (SolrDocument d : res.getResults()) {
                        String text = d.getFieldValue("status_text").toString();
                        if (text.toLowerCase().startsWith("rt "))
                            retweet++;
                        else
                            newTweet++;

                        if (d.getFieldValue("in_reply_to_screen_name") != null &&
                                d.getFieldValue("in_reply_to_screen_name").toString().length() > 0)
                            reply++;
                        if (d.getFieldValue("quoted_status_id_str") != null &&
                                d.getFieldValue("quoted_status_id_str").toString().length() > 0)
                            quote++;
                        if (d.getFieldValue("favorite_count") != null)
                            favorited += Integer.valueOf(d.getFieldValue("favorite_count").toString());
                        if (d.getFieldValue("reply_count") != null)
                            replied += Integer.valueOf(d.getFieldValue("reply_count").toString());
                        if (d.getFieldValue("retweet_count") != null)
                            retweeted += Integer.valueOf(d.getFieldValue("retweet_count").toString());
                        if (d.getFieldValue("quote_count") != null)
                            quoted += Integer.valueOf(d.getFieldValue("quote_count").toString());
                        if (d.getFieldValues("entities_hashtag") != null) {
                            for (Object o : d.getFieldValues("entities_hashtag"))
                                hashtags.add(o.toString());
                        }
                        if (d.getFieldValues("entities_symbol") != null) {
                            for (Object o : d.getFieldValues("entities_symbol"))
                                symbols.add(o.toString());
                        }
                        if (d.getFieldValues("entities_url") != null) {
                            for (Object o : d.getFieldValues("entities_url"))
                                urls.add(o.toString());
                        }
                        if (d.getFieldValues("entities_user_mention") != null) {
                            for (Object o : d.getFieldValues("entities_user_mention"))
                                mentions.add(o.toString());
                        }
                        if (d.getFieldValues("entities_media_url") != null) {
                            for (Object o : d.getFieldValues("entities_media_url"))
                                mediaURLs.add(o.toString());
                        }
                        if (d.getFieldValues("entities_media_type") != null) {
                            for (Object o : d.getFieldValues("entities_media_type"))
                                mediaTypes.add(o.toString());
                        }
                    }
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

        newDoc.addField("user_newtweet_count", newTweet);
        newDoc.addField("user_retweet_count", retweet);
        newDoc.addField("user_reply_count", reply);
        newDoc.addField("user_quote_count", quote);
        newDoc.addField("user_favorited_count", favorited);
        newDoc.addField("user_replied_count", replied);
        newDoc.addField("user_retweeted_count", retweeted);
        newDoc.addField("user_quoted_count", quoted);
        newDoc.addField("user_entities_hashtag", hashtags);
        newDoc.addField("user_entities_symbol", symbols);
        newDoc.addField("user_entities_url", urls);
        newDoc.addField("user_entities_user_mention", mentions);
        newDoc.addField("user_entities_media_url", mediaURLs);
        newDoc.addField("user_entities_media_type", mediaTypes);
    }

    /**
     * <field name="user_name" type="string" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_screen_name" type="string" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_statuses_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_friends_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_followers_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_listed_count" type="int" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_location" type="string" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_favorites_count" type="string" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_desc" type="text_general" indexed="true" stored="true" multiValued="false"/>
     * <field name="user_url" type="text_general" indexed="true" stored="true" multiValued="false"/>
     * <field name="text" type="text_general" indexed="true" stored="true" multiValued="false"/>
     * <field name="profile_background_image_url" type="string" indexed="true" stored="true" multiValued="false"/>
     * <field name="profile_image_url" type="string" indexed="true" stored="true" multiValued="false"/>
     *
     * @param userID
     * @param newDoc
     * @param userData
     */
    private void populateFromTwitterAPI(String userID, SolrInputDocument newDoc, Map<String, User> userData) {
        User u = userData.get(userID);
        if (u == null)
            LOG.warn(String.format("\t\tuser %s no longer exists!", userID));
        else {
            newDoc.addField("user_name", u.getName());
            newDoc.addField("user_screen_name", u.getScreenName());
            newDoc.addField("user_statuses_count", u.getStatusesCount());
            newDoc.addField("user_friends_count", u.getFriendsCount());
            newDoc.addField("user_followers_count", u.getFollowersCount());
            newDoc.addField("user_listed_count", u.getListedCount());
            newDoc.addField("user_location", u.getLocation());
            newDoc.addField("user_desc", u.getDescription());
            newDoc.addField("user_url", u.getURL());
            newDoc.addField("profile_background_image_url", u.getProfileBackgroundImageURL());
            newDoc.addField("profile_image_url", u.getProfileImageURL());
        }
    }

    private Map<String, User> makeTwitterLookupCall(List<String> userIDs) {
        Date now = new Date();
        /*long wait = (1200 - now.getTime() - prevCallDate.getTime());
        if (wait > 0) {*/
        //LOG.warn(String.format("\t\t API calling too frequent. Waiting %d milliseconds...", wait));
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
        //}
        prevCallDate = new Date();

        Map<String, User> result = new LinkedHashMap<>();
        try {
            long[] ids = new long[userIDs.size()];
            String[] idStr = userIDs.toArray(new String[0]);
            for (int i = 0; i < ids.length; i++)
                ids[i] = Long.valueOf(idStr[i]);
            ResponseList<User> users = twitter.lookupUsers(ids);

            //update each solr doc based on the twitter api results
            Map<String, User> tmp = new HashMap<>();
            for (User u : users)
                tmp.put(String.valueOf(u.getId()), u);
            for (String uid : userIDs)
                result.put(uid, tmp.get(uid));

        } catch (TwitterException e) {
            LOG.warn(String.format("\t\tunable to query Twitter, ids are: %s \n\t %s",
                    userIDs.toString(), ExceptionUtils.getFullStackTrace(e)));
        }
        return result;

    }


    /**
     * @param solrClient this must be the solr client on the 'tweets' core
     * @return
     */
    public static Set<String> getUniqueUsersFromSolr(SolrClient solrClient) {
        SolrQuery q = new SolrQuery();
        q.setQuery("*:*");
        q.setStart(0);
        q.setRows(10000);
        q.setFields("*");
        Set<String> userIDs = new HashSet<>();
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
                    for (SolrDocument d : res.getResults()) {
                        userIDs.add(d.getFieldValue("user_id_str").toString());
                    }
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
        return userIDs;
    }
}
