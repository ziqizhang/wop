package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.stats.user.UserStatsExtractor;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class ClusterUserFeatureGenerator {
    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(UserStatsExtractor.class.getName());

    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolrClient = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userSolrClient= new EmbeddedSolrServer(solrContainer.getCore("users"));

        ClusterUserFeatureGenerator ufg = new ClusterUserFeatureGenerator();
        ufg.process(args[1],args[2],tweetSolrClient, userSolrClient);

        tweetSolrClient.close();
        userSolrClient.close();

        System.exit(0);
    }




    private void process(String hashtagFile, String outFile,
                         SolrClient tweetCore, SolrClient userCore) throws IOException {
        Map<String, String> tag2disease = Common.createInverseHashtagMap(hashtagFile);
        LOG.info("Calculating users in multiple disease communities...");
        Pair<Set<String>, Map<String, String>> selected = findDiseasesAndUsers(userCore, tag2disease);
        List<String> orderedDiseases = new ArrayList<>(selected.getKey());
        List<String> orderedUsers = new ArrayList<>(selected.getValue().keySet());
        Collections.sort(orderedDiseases);
        Collections.sort(orderedUsers);

        LOG.info("\nCalculating feature matrix...");
        Matrix m = generateFeatures(tweetCore, orderedDiseases, orderedUsers,
                selected.getRight(), tag2disease);

        LOG.info("\nOutputing data...");
        CSVWriter csvWriter = null;

        csvWriter = Util.createCSVWriter(outFile);
        String[] header = new String[orderedUsers.size()+1];
        header[0]="";
        for(int i=0; i<orderedUsers.size(); i++)
            header[i+1]=selected.getRight().get(orderedUsers.get(i));
        csvWriter.writeNext(header);
        for (int r=0; r<m.numRows();r++){
            String[] row = new String[orderedUsers.size()+1];
            row[0]=orderedDiseases.get(r);
            for(int c=0;c<m.numColumns(); c++)
                row[c]=String.valueOf(m.get(r,c));
            csvWriter.writeNext(row);
        }
        csvWriter.close();
    }

    private Matrix generateFeatures(SolrClient tweetCore,
                                    List<String> orderedDiseases,
                                    List<String> orderedUsers,
                                    Map<String, String> selectedUsers,
                                    Map<String, String> tag2Disease) {
        Matrix m = new LinkedSparseMatrix(orderedDiseases.size(), selectedUsers.size());

        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(resultBatchSize);
        long tweets = 0;
        boolean stop = false;
        while (!stop) {
            QueryResponse qr = null;
            try {
                qr = Util.performQuery(query, tweetCore);
                if (qr != null)
                    tweets = qr.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        tweets, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {
                    String userid = d.getFieldValue("user_id_str").toString();
                    if (!orderedUsers.contains(userid))
                        continue;
                    Collection<Object> hashtags = d.getFieldValues("entities_hashtag");
                    if(hashtags==null)
                        continue;
                    for (Object h : hashtags) {
                        String di = tag2Disease.get(h.toString());
                        if (di == null)
                            continue;

                        int diseaseIndex = orderedDiseases.indexOf(di);
                        if (diseaseIndex < 0)
                            continue;

                        int userIndex = orderedUsers.indexOf(userid);
                        m.set(diseaseIndex, userIndex, m.get(diseaseIndex, userIndex) + 1);
                    }

                }
            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < tweets)
                query.setStart(curr);
            else
                stop = true;
        }
        return m;
    }

    /**
     * find disease tags that have overlapping users, and also these users
     */
    private Pair<Set<String>, Map<String, String>> findDiseasesAndUsers(SolrClient userCore, Map<String, String> tag2Disease) {
        Map<String, String> selectedUsers = new HashMap<>();
        Set<String> selectedDiseases = new HashSet<>();

        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(resultBatchSize);
        long users = 0;
        boolean stop = false;
        while (!stop) {
            QueryResponse qr = null;
            try {
                qr = Util.performQuery(query, userCore);
                if (qr != null)
                    users = qr.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        users, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {

                    Collection<Object> hashtags = d.getFieldValues("user_entities_hashtag");
                    if (hashtags==null ||hashtags.size() == 1)
                        continue;

                    Set<String> diseases = new HashSet<>();
                    for (Object o : hashtags) {
                        String di = tag2Disease.get(o.toString());
                        if (di != null)
                            diseases.add(di);
                    }
                    if (diseases.size() < 2)
                        continue;

                    int tweets = Integer.valueOf(d.getFieldValue("user_newtweet_count").toString());
                    tweets += Integer.valueOf(d.getFieldValue("user_retweet_count").toString());
                    if (tweets < 2)
                        continue;

                    //if not, save user id, save disease id
                    Object username = d.getFieldValue("user_screen_name");
                    if (username==null)
                        username=d.getFieldValue("id");

                    String userid = d.getFieldValue("id").toString();
                    selectedUsers.put(userid, username.toString());
                    selectedDiseases.addAll(diseases);
                }
            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < users)
                query.setStart(curr);
            else
                stop = true;
        }

        return new ImmutablePair<>(selectedDiseases, selectedUsers);
    }


}
