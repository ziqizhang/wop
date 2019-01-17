package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import no.uib.cipr.matrix.Matrix;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.stats.user.UserStatsExtractor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

public class Common {

    /**
     * find disease tags that have overlapping users, and also these users
     */
    static Pair<Map<String, Set<String>>, Map<String, String>>
    findDiseasesAndUsers(SolrClient userCore, Map<String, String> tag2Disease, Logger logger, int resultBatchSize) {
        Map<String, String> userIDs = new HashMap<>();
        Map<String, Set<String>> userAndDiseases = new HashMap<>();

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
                logger.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        users, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {

                    Collection<Object> hashtags = d.getFieldValues("user_entities_hashtag");
                    if (hashtags==null)
                        continue;

                    Set<String> diseases = new HashSet<>();
                    for (Object o : hashtags) {
                        String di = tag2Disease.get(o.toString());
                        if (di != null)
                            diseases.add(di);
                    }


                    Object username = d.getFieldValue("user_screen_name");
                    if (username==null)
                        username=d.getFieldValue("id");

                    String userid = d.getFieldValue("id").toString();
                    userIDs.put(userid, username.toString());

                    Set<String> userInterestedDiseases = userAndDiseases.get(userid);
                    if (userInterestedDiseases==null)
                        userInterestedDiseases=new HashSet<>();
                    userInterestedDiseases.addAll(diseases);
                    userAndDiseases.put(userid, userInterestedDiseases);
                }
            } catch (Exception e) {
                logger.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < users)
                query.setStart(curr);
            else
                stop = true;
        }

        return new ImmutablePair<>(userAndDiseases, userIDs);
    }

    /**
     * find disease tags that have overlapping users, and also these users
     */
    static Pair<Map<String, Long>, Long>
    findHashtagsAndUsers(SolrClient userCore, Logger logger, int resultBatchSize) {
        Map<String, Long> tagAndUsercount = new HashMap<>();

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
                logger.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        users, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {

                    String userid = d.getFieldValue("id").toString();
                    Collection<Object> hashtags = d.getFieldValues("user_entities_hashtag");
                    if (hashtags==null)
                        continue;

                    for (Object o : hashtags) {
                        String tag = o.toString();
                        if (tag.length()>1){
                            Long userC= tagAndUsercount.get(tag);
                            if (userC==null)
                                userC=0L;
                            userC++;
                            tagAndUsercount.put(tag, userC);
                        }
                    }


                }
            } catch (Exception e) {
                logger.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < users)
                query.setStart(curr);
            else
                stop = true;
        }

        return new ImmutablePair<>(tagAndUsercount,users);
    }

    /**
     * find hashtags used with different disease communities
     *
     * @return key: hashtag; value: a set of disease ID tags for communities
     */
    static Map<String, Set<String>> findTagsAndDiseases(SolrClient tweetCore, Map<String, String> tag2diseaseInput,
                                                               Logger logger, int resultBatchSize) {
        Map<String, Set<String>> tag2diseases=new HashMap<>();

        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(resultBatchSize);
        long users = 0;
        boolean stop = false;
        while (!stop) {
            QueryResponse qr = null;
            try {
                qr = Util.performQuery(query, tweetCore);
                if (qr != null)
                    users = qr.getResults().getNumFound();
                //update results
                logger.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        users, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {
                    Collection<Object> hashtagsO = d.getFieldValues("entities_hashtag");
                    Set<String> hashtags =new HashSet<>();
                    if (hashtagsO==null ||hashtagsO.size() == 1)
                        continue;

                    Set<String> currDiseases = new HashSet<>();
                    for (Object o : hashtagsO) {
                        String tag = o.toString();
                        if (tag.length()<2)
                            continue;
                        hashtags.add(tag);
                        String di = tag2diseaseInput.get(tag);
                        if (di != null)
                            currDiseases.add(di);
                    }
                    if(currDiseases.size()<2)
                        continue;
                    for (String tag: hashtags){
                        Set<String> allDiseases=tag2diseases.get(tag);
                        if(allDiseases==null)
                            allDiseases=new HashSet<>();
                        allDiseases.addAll(currDiseases);
                        tag2diseases.put(tag,allDiseases);
                    }

                }
            } catch (Exception e) {
                logger.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < users)
                query.setStart(curr);
            else
                stop = true;
        }

        return tag2diseases;
    }

    /**
     * @return key: hashtag; value: disease tag id
     */
    static Map<String, String> createInverseHashtagMap(String hashtagFile) throws IOException {
        Map<String, List<String>> hashtags = Util.readHashtags(new File(hashtagFile));
        Map<String, String> res = new HashMap<>();
        for (Map.Entry<String, List<String>> en : hashtags.entrySet()) {
            for (String tag : en.getValue()) {
                res.put(tag, en.getKey());
            }
        }
        return res;
    }

    static void saveMatrixData(Matrix m, List<String> matrixIndex, String outFile) throws IOException {
        CSVWriter csvWriter = null;

        csvWriter = Util.createCSVWriter(outFile);
        String[] header = new String[matrixIndex.size()+1];
        header[0]="";
        for(int i=0; i<matrixIndex.size(); i++)
            header[i+1]=trimHashChar(matrixIndex.get(i));
        csvWriter.writeNext(header);
        for (int r=0; r<m.numRows();r++){
            String[] row = new String[matrixIndex.size()+1];
            row[0]=trimHashChar(matrixIndex.get(r));
            for(int c=0;c<m.numColumns(); c++)
                row[c+1]=String.valueOf(m.get(r,c));
            csvWriter.writeNext(row);
        }
        csvWriter.close();
    }

    static void saveCommunityPairSimilarityData(Matrix m, List<String> matrixIndex, String outFile) throws IOException {
        CSVWriter csvWriter = null;

        csvWriter = Util.createCSVWriter(outFile);
        String[] header = new String[3];
        header[0]="DiseaseCommunity1";
        header[1]="DiseaseCommunity2";
        header[2]="SimilarityScore(Dice)";

        for(int i=0; i<matrixIndex.size();i++){
            String d1 = matrixIndex.get(i);
            for(int j=i+1; j<matrixIndex.size();j++){
                String d2 =matrixIndex.get(j);
                double sim = m.get(i,j);
                if(sim>0){
                    String[] row =new String[3];
                    row[0]=d1;
                    row[1]=d2;
                    row[2]=String.valueOf(sim);
                    csvWriter.writeNext(row);
                }

            }
        }

        csvWriter.close();
    }

    static String trimHashChar(String s){
        if(s.startsWith("#"))
            return s.substring(1).trim();
        return s;
    }

    static Set<String> readExcludeList(String csvFile, int max) throws IOException {
        List<String> lines=FileUtils.readLines(new File(csvFile), Charset.forName("utf-8"));
        Set<String> exclusion=new HashSet<>();
        boolean foundStart=false;
        int count=0;
        for(String l : lines){
            if (l.startsWith("OUTLIERS")){
                foundStart=true;
                continue;
            }
            if(foundStart){
                String[] parts = l.split(",");
                if(count>=max||l.equalsIgnoreCase(""))
                    break;

                exclusion.add(parts[0].trim());
                count++;
            }
        }
        return exclusion;
    }
}
