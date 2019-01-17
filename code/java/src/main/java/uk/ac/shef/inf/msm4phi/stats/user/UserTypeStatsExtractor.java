package uk.ac.shef.inf.msm4phi.stats.user;

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
import java.util.List;
import java.util.Map;

/**
 * stakeholder counts for each disease
 */

public class UserTypeStatsExtractor extends IndexAnalyserWorker {
    private static final Logger LOG = Logger.getLogger(UserTypeStatsExtractor.class.getName());


    public UserTypeStatsExtractor(int id, SolrClient solrClient, String outFolder) {
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

                long users = 0, patients=0, hpi=0, hpo = 0,
                advocate= 0, research = 0, other = 0, none=0;

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
                            //label
                            Map<String, Object> fieldValues=d.getFieldValueMap();
                            if (fieldValues.containsKey("label_s")){
                                String label=fieldValues.get("label_s").toString();
                                if (label.equalsIgnoreCase("patient"))
                                    patients++;
                                else if(label.equalsIgnoreCase("hpo"))
                                    hpo++;
                                else if(label.equalsIgnoreCase("hpi"))
                                    hpi++;
                                else if(label.equalsIgnoreCase("advocate"))
                                    advocate++;
                                else if(label.equalsIgnoreCase("research"))
                                    research++;
                                else if(label.equalsIgnoreCase("other"))
                                    other++;
                            }
                            else
                                none++;
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

                //prepare line to write to csv
                String[] line = new String[9];
                line[0] = en.getKey();
                line[1] = String.valueOf(users);
                line[2] = String.valueOf(patients);
                line[3] = String.valueOf(advocate);
                line[4] = String.valueOf(research);
                line[5] = String.valueOf(hpi);
                line[6] = String.valueOf(hpo);
                line[7] = String.valueOf(other);
                line[8] = String.valueOf(none);
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
     * @param csvWriter
     */
    private void writeCSVHeader(CSVWriter csvWriter) {
        //0         1       2
        String[] headerRecord = {"tag", "unique_users", "patients",
                //3             4               5              6               7
                "advocate", "research", "hpi", "%hpo", "other",
                //8                9                 10         11
                "none"};
        csvWriter.writeNext(headerRecord);
    }

}

