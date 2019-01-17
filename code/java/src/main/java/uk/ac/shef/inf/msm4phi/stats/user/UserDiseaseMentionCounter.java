package uk.ac.shef.inf.msm4phi.stats.user;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * given a list of users and a disease hashtag mapping,
 * count how many diseases a user have mentioned in the indexed data
 *
 * (does not use api)
 */
public class UserDiseaseMentionCounter {

    public static void main(String[] args) throws IOException {
        SolrClient solrClient= Util.getSolrClient(Paths.get(args[0]),"tweets");
        UserDiseaseMentionCounter processor = new UserDiseaseMentionCounter(solrClient);
        List<String[]> users=processor.readUsers(args[1],0,1,21);
        Map<String, String> diseases=processor.loadHashtagDiseaseMap(args[2]);
        processor.createFeatures(users, diseases, args[3]);
        System.exit(0);
    }

    private static final Logger LOG = Logger.getLogger(UserDiseaseMentionCounter.class.getName());

    private SolrClient solrTweet;

    public UserDiseaseMentionCounter(SolrClient solrTweet){
        this.solrTweet=solrTweet;
    }

    public void createFeatures(List<String[]> users, Map<String, String> hashtagToDisease,
                               String outFile) throws IOException {
        List<String[]> output=new ArrayList<>();

        for(String[] user : users) {
            Map<String, Integer> diseaseCounts=new HashMap<>();
            String userID=user[1];
            SolrQuery q = Util.createQueryTweetsOfUserScreenname(10000, userID);
            boolean stop = false;

            while (!stop) {
                QueryResponse res = null;
                long total = 0;
                try {
                    res = Util.performQuery(q, solrTweet);
                    if (res != null) {
                        total = res.getResults().getNumFound();
                        //update results
                        LOG.info(String.format("\t\ttotal tweets of user %s is %d, currently processing from %d to %d...",
                                userID, total, q.getStart(), q.getStart() + q.getRows()));
                        for (SolrDocument d : res.getResults()) {
                            if (d.getFieldValues("entities_hashtag") != null) {
                                for (Object o : d.getFieldValues("entities_hashtag")) {
                                    String tag = o.toString();
                                    String disease=hashtagToDisease.get(tag);
                                    if (disease==null)
                                        continue;
                                    Integer count=diseaseCounts.get(disease);
                                    count=count==null?1:count+1;
                                    diseaseCounts.put(disease,count);
                                }
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
            String[] row = new String[3];
            row[0]=user[0];
            row[1]=userID;
            row[2]=String.valueOf(diseaseCounts.size());
            output.add(row);
        }

        Writer writer = Files.newBufferedWriter(Paths.get(outFile));

        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.NO_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);
        String[] header=new String[3];
        header[0]="user_id";
        header[1]="user_screenname";
        header[2]="disease_count";
        csvWriter.writeNext(header);
        for (String[] row :output){
            csvWriter.writeNext(row);
        }
        csvWriter.close();
    }

    private List<String[]> readUsers(String inputCSV, int colID, int colScreename,
                                     int colHashtag) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(inputCSV));
        CSVReader csvReader = new CSVReader(reader);

        // Reading Records One by One in a String array
        List<String[]> res = new ArrayList<>();
        String[] nextRecord;
        int count=0;
        while ((nextRecord = csvReader.readNext()) != null) {
            if (count==0) {
                count++;
                continue;
            }
            String[] row = new String[5];
            row[0]=nextRecord[colID];
            row[1]=row[0].substring(row[0].lastIndexOf("/")+1);
            row[2]=nextRecord[colHashtag];
            row[3]="0";
            res.add(row);
        }
        return res;
    }

    /**
     *
     * @param hashtagCSVFile containing 2 columns col1=hashtag, col2=disease id
     * @return
     */
    private Map<String, String> loadHashtagDiseaseMap(String hashtagCSVFile) throws IOException {
        Map<String, String> output = new HashMap<>();
        Reader reader = Files.newBufferedReader(Paths.get(hashtagCSVFile));
        CSVReader csvReader = new CSVReader(reader);

        // Reading Records One by One in a String array
        List<String[]> res = new ArrayList<>();
        String[] nextRecord;
        int count=0;
        while ((nextRecord = csvReader.readNext()) != null) {
            if (count == 0) {
                count++;
                continue;
            }

            output.put(nextRecord[0].toLowerCase().substring(1),nextRecord[1]);
        }
        return output;
    }
}
