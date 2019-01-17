package uk.ac.shef.inf.msm4phi.stats.user;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import twitter4j.Paging;
import twitter4j.ResponseList;
import twitter4j.Status;
import twitter4j.Twitter;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * reads in the 'basic_feature' csv file, if the user profile is empty, collect most recent
 * 20 tweets and concatenate them as their profile
 */
public class UserProfileFiller {

    public static void main(String[] args) throws IOException {
        UserProfileFiller utc = new UserProfileFiller(
                Integer.valueOf(args[0]),
                args[1],args[2],args[3],args[4]
        );

        File folder = new File(args[5]);
        for (File f : folder.listFiles()) {
            System.out.println(f);
            utc.process(f.toString(), 0, 14, 16,
                    args[6]);
        }
    }

    private static final Logger LOG = Logger.getLogger(UserProfileFiller.class.getName());
    private Twitter twitter;
    private int tweetCount;

    public UserProfileFiller(int tweetCount,
                                    String twitterConsumerKey, String twitterConsumerSecret,
                                    String twitterAccessToken, String twitterAccessSecret){
        this.twitter = Util.authenticateTwitter(twitterConsumerKey, twitterConsumerSecret, twitterAccessToken,
                twitterAccessSecret);
        this.tweetCount=tweetCount;
    }

    public void process(String inputCSV, int colID, int colScreename,
                        int colProfile,
                        String outFolder) throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get(inputCSV));
        CSVReader csvReader = new CSVReader(reader);

        String filename = inputCSV.substring(inputCSV.lastIndexOf("/") + 1);
        Writer writer = Files.newBufferedWriter(Paths.get(outFolder + "/" + filename));
        CSVWriter csvWriter = new CSVWriter(writer,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.NO_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        // Reading Records One by One in a String array
        String[] nextRecord;
        int count = 0, countCollected=0;

        while ((nextRecord = csvReader.readNext()) != null) {
            if (count == 0) {
                count++;
                csvWriter.writeNext(nextRecord);
                continue;
            }

            String profile = nextRecord[colProfile];
            if (profile.trim().length() < 2) {
                countCollected++;
                String filledProfile = fillProfile(nextRecord[colScreename], countCollected);
                nextRecord[colProfile] = filledProfile;
            }

            csvWriter.writeNext(nextRecord);
        }
        csvReader.close();
        csvWriter.close();
    }

    private String fillProfile(String screenname, int countU){
        LOG.info(String.format("processing user %s, %s",
                screenname, String.valueOf(countU)));

        int page=1, total_added=0;
        String profile="";
        while(true) {
            LOG.info(String.format("\t\tpage %s",
                    page));
            Paging paging = new Paging(page, tweetCount);
            ResponseList<Status> statuses = null;
            try {
                statuses = twitter.getUserTimeline(screenname, paging);
                Thread.sleep(1000);
            } catch (Exception e) {
                LOG.warn(String.format("\t\tfailed to count user tweets %s \n\t %s, trying again in 10 secs...",
                        screenname, ExceptionUtils.getFullStackTrace(e)));
                try {
                    Thread.sleep(10000);
                    break;
                } catch (InterruptedException e1) {
                }
            }

            page++;


            for(Status s :statuses){
                profile+=s.getText().replaceAll("\\s+"," ").trim()+" ";
                total_added++;
                if (total_added==tweetCount)
                    break;
            }

            if(total_added==tweetCount||statuses.size()==0)
                break;
        }

        return profile.trim();
    }
}
