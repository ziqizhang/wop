package uk.ac.shef.inf.wop;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringEscapeUtils;
import org.jsoup.Jsoup;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Util {

    private static long tooMany=0;
    private static long tooFew=0;
    public static String numeric="\\b([0-9]+.[0-9]+)|([0-9]+)\\b";
    public static Pattern numericP = Pattern.compile(numeric);
    public static String alphanum="(?=[A-Za-z,.]*\\d+[A-Za-z,.]*)(?=[\\d,.]*[A-Za-z]+[\\d,.]*)[A-Za-z\\d,.]{2,}(?<![,.])";
    public static Pattern alphanumP=Pattern.compile(alphanum);

    private static String toASCII(String in) {
        String fold = in.replaceAll("[^\\p{ASCII}]", "").
                replaceAll("\\s+", " ").trim();
        return fold;
    }

    public static int replacePatterns(String in, Pattern pat){
        Matcher m = pat.matcher(in);
        int found=0;
        while(m.find()) {
            found++;
            //System.out.println(in.substring(m.start(),m.end()));
        }
        return found;
    }

    private static String cleanDesc(String value, boolean lower) {
        value = Jsoup.parse(value).text();
        try {
            value = StringEscapeUtils.unescapeJava(value);
            if (lower)
                value=value.toLowerCase();
        }catch (Exception e){
            System.out.println("\t"+value);
        }

        //String asciiValue = toASCII("M5 x 35mm Full Thread Hexagon Bolts (DIN 933) - PEEK DescriptionThe M5 x 35mm Full Thread Hexagon Bolts (DIN 933) - PEEK has the following features:M5 (5mm) Thread Size (T)DIN 933 Manufacturing Standard35mm Length (L)Yes Fully ThreadedPEEK MaterialPEEK Thermoplastic Material Specification0.8mm Pitch3.6mm Head Length (K)8mm Head Width A/F (H)Self Colour Finish+/- 0.13mm General Tolerance");
        String asciiValue = toASCII(value);

        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}:,.;]", " ").
                replaceAll("\\s+", " ").trim();
        //value= StringUtils.stripAccents(value);

        int nums = replacePatterns(alphanumeric,numericP);
        int an= replacePatterns(alphanumeric, alphanumP);

        String alphanumeric_clean = alphanumeric.replaceAll(alphanum,"LETTERNUMBER");
        alphanumeric_clean = alphanumeric_clean.replaceAll(numeric,"NUMBER");


        List<String> normTokens = Arrays.asList(alphanumeric_clean.split("\\s+"));
        if ((nums+an)>=(normTokens.size()/3)){
            System.out.println("> too many num/numdigit tokens"+(nums+an)+"/"+normTokens.size());
            return null;
        }

        if (normTokens.size() > 200) {
            tooMany++;
            System.out.println("> "+normTokens.size());
            return null;
        }
        if (normTokens.size() <5) {
            tooFew++;
            return null;
        }
        return alphanumeric_clean;
    }
    private static void filterDescriptions(String inFile, String outFile){

        try (BufferedReader br = new BufferedReader(new FileReader(inFile))) {
            PrintWriter p = new PrintWriter(outFile);

            long count=0;
            String line;
            while ((line = br.readLine()) != null) {
                line=cleanDesc(line,true);
                if (line==null)
                    continue;

                count++;
                p.println(line);
                if (count%100000==0)
                    System.out.println(count);
            }

            p.close();
            System.out.println(count);
            System.out.println("too many="+tooMany);
            System.out.println("too few="+tooFew);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void filterNames(String inFile, String outFile) throws IOException {
        List<String> names= FileUtils.readLines(new File(inFile));
        PrintWriter p = new PrintWriter(outFile);
        for (String n : names){
            n=n.toLowerCase();
            String alphanumeric = n.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}:,.;]", " ").
                    replaceAll("\\s+", " ").trim();

            String alphanumeric_clean = alphanumeric.replaceAll(alphanum,"LETTERNUMBER");
            alphanumeric_clean = alphanumeric_clean.replaceAll(numeric,"NUMBER");
            p.println(alphanumeric_clean);
        }

    }

    public static void main(String[] args) throws IOException {
        //filterDescriptions(args[0],args[1]);
        filterNames(
                "/home/zz/Work/data/mt/product/translation_in/goldstandard_eng_v1_utf8_names_casesensitive.txt",
                "/home/zz/Work/data/mt/product/translation_in/goldstandard_eng_v1_utf8_names.txt");
    }
}
