#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

//ROOT include files
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

//Fastjet headers
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include <fastjet/tools/JHTopTagger.hh>
#include <fastjet/Selector.hh>

//Boost headers
#include <boost/algorithm/string.hpp>
#include <boost/tuple/tuple.hpp>

//custom headers
#include "TopHist.h"
#include "complex_d.h"

using namespace std;
using namespace fastjet;

//----------------------------------------------------------------------
// Some four-vector operators
//----------------------------------------------------------------------
double dot(fastjet::PseudoJet p1, fastjet::PseudoJet p2);
double deltaR(fastjet::PseudoJet p1, fastjet::PseudoJet p2);

/* jet to lepton mistag */
double Pjet_to_lepton(double pt);

/* jet to photon mistag */
double Pjet_to_photon(double pt);

//----------------------------------------------------------------------// forward declaration for printing out info about a jet
//----------------------------------------------------------------------
ostream & operator<<(ostream &, const PseudoJet &);

//----------------------------------------------------------------------
// command line parameters
//----------------------------------------------------------------------
char* getCmdOption(char ** begin, char ** end, const std::string & option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

//----------------------------------------------------------------------
// Analysis functions
//----------------------------------------------------------------------

// smearing of jets, leptons and photons. 
fastjet::PseudoJet smear_jet(fastjet::PseudoJet jet_in);
fastjet::PseudoJet smear_lepton(fastjet::PseudoJet lepton_in, int lepton_id);
fastjet::PseudoJet smear_photon(fastjet::PseudoJet photon_in);

// acceptance efficiency for leptons, photons, jets
bool lepton_efficiency_accept(fastjet::PseudoJet lepton_in, int lepton_id);
bool photon_efficiency_accept(fastjet::PseudoJet photon_in);
bool jet_efficiency_accept(fastjet::PseudoJet jet_in);

// IDs of B-hadrons used by the btag_hadrons function
int bhadronid[105] = {5122, -5122, 15122, -15122, 5124, -5124, 5334, -5334, 5114, -5114, 5214, -5214, 5224, -5224, 5112, -5112, 5212, -5212, 5222, -5222, 15322, -15322, 15312, -15312, 15324, -15324, 15314, -15314, 5314, -5314, 5324, -5324, 5132, -5132, 5232, -5232, 5312, -5312, 5322, -5322, 551, 10555, 100551, 200551, 553, 557, 555, 100555, 200555, 20523, -20523, 20513, -20513, 20543, -20543, 20533, -20533, 511, 521, -511, -521, 531, -531, 541, -541, 513, 523, -513, -523, 533, -533, 543, -543, 10513, 10523, -10513, -10523, 10533, -10533, 10543, -10543, 10511, 10521, -10511, -10521, 10531, -10531, 10541, -10541, 20513, 20523, -20513, -20523, 20533, -20533, 20543, -20543, 515, 525, -515, -525, 535, -535, 545, -545};

/*
 * CREATE ROOT CHAIN TO READ IN THE FILES
 */
TChain t("Data");


/* 
 * DECLARE RANDOM NUMBERS
 */ 
TRandom3 rnd;
TRandom3 rndint;

/***** 
 ***** SWITCHES FOR SMEARING/EFFICIENCIES
 *****/
bool donotsmear_jets = 0;
bool donotsmear_leptons = 0;
bool donot_apply_efficiency = 0;
bool donotsmear_photons = 0;

int main(int argc, char *argv[]) {

  //take command line options
  char* output;
  char* infile = "";
  if(argv[1]) { infile = argv[1]; } else { cout << "Use: ./HwSimAnalysis [input] [options]" << endl; exit(1); }

  //set the variables and addresses to be read from root file
  //total number of particles in an event
  int numparticles;
 
  /** particle information in the order: 
   * 4 momenta (E,x,y,z), id, other info
   **/
  double objects[8][10000];
  /* the event weight */ 
  double evweight;
  /* The missing energy four-vector */
  double theETmiss[4];

  /* parton-level particle information (final state)
   * 4 momenta (E,x,y,z), id 
   */
  double partons[5][100];
  
  /* parton-level particle information (INCOMING)
   * 4 momenta (E,x,y,z), id
   */
  double incoming[5][2];
  
  //total number of outgoing partons in hard process
  int numoutgoing; 

  
  /* the optional weight values */
  std::vector<double> *theOptWeights;

  /* the optional weight names */
  std::vector<string> *theOptWeightsNames;


  /* 
   * SET THE ROOT BRANCH ADDRESSES
   */
  t.SetBranchAddress("numparticles",&numparticles);
  t.SetBranchAddress("objects",&objects);
  t.SetBranchAddress("evweight",&evweight);
       
  t.SetBranchAddress("theETmiss", &theETmiss);
  
  t.SetBranchAddress("numoutgoing", &numoutgoing);
  t.SetBranchAddress("partons", &partons);
  t.SetBranchAddress("incoming", &incoming);


    
  // uncomment if you wish to use optional weights:
  /* t.SetBranchAddress("theOptWeightsNames", &theOptWeightsNames);
     t.SetBranchAddress("theOptWeights", &theOptWeights);*/
        
  /* Set up random number
   * generator
   */ 
  rnd.SetSeed(14101983);

  /* Add up all the input 
   * files to the chain
   */ 
  string stringin = "";
  ifstream inputlist;
  if (std::string(infile).find(".input") != std::string::npos) {
    inputlist.open(infile);
    if(!inputlist) {  cerr << "Error: Failed to open input file " << infile << endl; exit(1); }
    while(inputlist) { 
      inputlist >> stringin; 
      if(stringin!="") { t.Add(stringin.c_str()); 
        cout << "Adding " << stringin.c_str() << endl;
      }
      stringin = "";
    }
    inputlist.close();
  } else if (std::string(infile).find(".root") != std::string::npos) {
    cout << "Adding " << infile << endl;
    t.Add(infile);
  }

  /* Get Number of events
   * and print
   */
  int EventNumber(int(t.GetEntries()));
  cout << "Total number of events in " << infile << " : " << EventNumber << endl;

  /* 
   * -b: USED TO REANALYZE PREVIOUSLY PASSED EVENTS ONLY, DEFAULT IS ALL EVENTS 
   */
  
  //whether the analysis performed is level-2 or level-3 
  bool basic = true;
  if(cmdOptionExists(argv, argv+argc, "-b")) {
    cout << "Looking for .evp2 file, running over all events" << endl;
    basic = false;
  }


  /* 
   * -t: ADD AN EXTENSION TAG TO YOUR OUTPUT FILES
   */
  string tag;
  tag = "";
  if(cmdOptionExists(argv, argv+argc, "-t")) {
    tag = getCmdOption(argv, argv + argc, "-t");  
    cout << "Adding tag: " << tag << endl;
    tag = "-" + tag;
  }

  /* 
   * -n: RUN FROM START OF FILE UP TO A GIVEN NUMBER OF EVENTS
   */
  char * switch_maxevents;
  char * switch_minevents;
  int maxevents(0), minevents(0);
  if(cmdOptionExists(argv, argv+argc, "-n")) {
    switch_maxevents = getCmdOption(argv, argv + argc, "-n");  
    maxevents=(atoi(switch_maxevents));	       
    if(maxevents > EventNumber) { maxevents = EventNumber; } 
    cout << "Analyzing up to " << maxevents << endl;
    if(maxevents < 1 || maxevents > 1E10) { cout << "Error: maxevents must be in the range [1,1E10]" << endl; exit(1); } 
  }
  
  /* 
   * -nmax: RUN FROM START OF FILE UP TO A GIVEN NUMBER OF EVENTS, TO BE USED IN CONJUNCTION WITH -nmin
   */
  //maximum number of events to analyze
  if(cmdOptionExists(argv, argv+argc, "-nmax") && !cmdOptionExists(argv, argv+argc, "-n")) {
   switch_maxevents = getCmdOption(argv, argv + argc, "-nmax");  
    maxevents=(atoi(switch_maxevents));	       
    if(maxevents > EventNumber) { maxevents = EventNumber; } 
    cout << "Analyzing up to " << maxevents << endl;
    if(maxevents < 1 || maxevents > 1E10) { cout << "Error: maxevents must be in the range [1,1E10]" << endl; exit(1); } 
  } 
  if(!cmdOptionExists(argv, argv+argc, "-nmax") && !cmdOptionExists(argv, argv+argc, "-n")) { maxevents = EventNumber; }

  /* 
   * -nmin: RUN FROM POINT nmin OF FILE UP TO A GIVEN NUMBER OF EVENTS SPECIFIED BY -nmax
   */
  //starting number of events to analyse
  if(cmdOptionExists(argv, argv+argc, "-nmin")) {
    switch_minevents = getCmdOption(argv, argv + argc, "-nmin");  
    minevents=(atoi(switch_minevents));	       
    if(minevents > maxevents) { minevents = 0; }
    cout << "Analyzing from " << minevents << endl;
    if(minevents < 1 || minevents > 1E10) { cout << "Error: minevents must be in the range [1,1E10]" << endl; exit(1); } 
  }


  /* 
   * CREATE THE OUTPUT FILE STRINGS 
   */
  string outnew = "";
  outnew = std::string(infile);
  string replacement = tag + ".top";
  boost::replace_all(outnew, ".root", replacement);
  boost::replace_all(outnew, ".input", replacement);        
  char* output2 = new char[outnew.length() + 1];
  //  cout << outnew.c_str() << endl;
  strcpy (output2, outnew.c_str());
  output = output2;
  
  char* output_dat;
  string outnew2 = "";
  outnew2 = std::string(infile);
  replacement = tag + ".dat";
  boost::replace_all(outnew2, ".root", replacement);
  boost::replace_all(outnew2, ".input", replacement);        
  char* output3 = new char[outnew2.length() + 1];
  strcpy (output3, outnew2.c_str());
  output_dat = output3;
  ofstream outdat(output_dat, ios::out);

  //load events that have passed the second stage of analysis
  //if basic = false;
  string ineventpass;
  ifstream inevt;
  string inevt_curr;
  int passed_event[20000];  
  int npassed_previous(0);
  if(basic == false) { 
    ineventpass = std::string(infile);
    replacement = tag + ".evp";
    boost::replace_all(ineventpass,".input", replacement);
    boost::replace_all(ineventpass,".root", replacement);
    inevt.open(ineventpass.c_str());
    if(!inevt) { cerr << "Error: Cannot open "<< ineventpass.c_str() << endl; exit(1); } 
    for(int ii = 0; ii < 1000; ii++) { passed_event[ii] = -1; }
    while(inevt) { 
      inevt >> inevt_curr;
      // cout << inevt_curr.c_str() << endl;
      passed_event[npassed_previous] = atoi(inevt_curr.c_str());
      npassed_previous++;
    }
  }
  //for(int pp = 0; pp < npassed_previous; pp++) { coust << passed_event[pp] << endl; }
 
  string outeventpass = ""; 
  ofstream outevp;

  if(basic == false) { 
    outeventpass = std::string(infile);
    replacement = tag + ".evp2";
    boost::replace_all(outeventpass,".root", replacement);
    boost::replace_all(outeventpass,".input", replacement);
    boost::replace_all(outeventpass,".top", replacement);
    outevp.open(outeventpass.c_str());
  } else if(basic == true) {
    outeventpass = std::string(infile);
    replacement = tag + ".evp";
    boost::replace_all(outeventpass,".root", replacement);
    boost::replace_all(outeventpass,".input", replacement);
    boost::replace_all(outeventpass,".top", replacement);
    outevp.open(outeventpass.c_str());
  }

  /*
   * PREPARES THE OUTPUT ARRAY FOR *_var.root: USED FOR FURTHER ANALYSIS
   */
  std::cout << "Preparing Root Tree for event variables" << endl;
  TTree* Data2;
  TFile* dat2;
  string fnameroot = std::string(infile);
  replacement = tag + "_var.root";
  boost::replace_all(fnameroot,".root", replacement);
  boost::replace_all(fnameroot,".input", replacement);
  dat2 = new TFile(fnameroot.c_str(), "RECREATE");
  Data2 = new TTree ("Data2", "Data Tree");
  //variables to fill in the .root file
  double variables[10]; 
  double eventweight[1];
  double muonevent[1];
  Data2->Branch("variables", &variables, "variables[10]/D");
  Data2->Branch("eventweight", &eventweight, "eventweight[1]/D");

   /* particles to exclude from jet finder:
   * neutrinos
   */
  int excluded[6] = { 12, -12, 14, -14, 16, -16 } ;
  
  vector<int> exclude_ids;
  for(int ex = 0; ex < 6; ex++) exclude_ids.push_back(excluded[ex]);
						
  fastjet::PseudoJet pcurr; //the current particle under consideration
  vector<fastjet::PseudoJet> input_particles; //the input particles to the jet finder

  /* 
   * CUTS DEFINED HERE IN GEV
   */
  
  /* ALL PARTICLE CUTS */
  double cut_eta(5.0); //global pseudorapidity cut of particles
  double cut_pt_part(0.1); //global pt cut for particles

  /* JET CUTS */
  double cut_pt_jet(20.0); //pt cut for jets
  double cut_eta_jet(5.0); //pseudo-rapidity cut for jets

  /* LEPTON CUTS */
  double cut_eta_electron(3.0); // pseudo-rapidity cut for electrons
  double cut_pt_electron(10.0); // pt cut for electrons

  double cut_eta_muon(3.0); // pseudo-rapidity cut for muons
  double cut_pt_muon(10.0); // pt cut for muons

  /* PHOTON CUTS */
  double cut_pt_photon(20.0); //pt cut for jets
  double cut_eta_photon(5.0); //pseudo-rapidity cut for jets

   /*
   * jet algorithm R
   */
  double R=0.4;
  /* 
   * jet algorithm details
   */ 
  fastjet::RecombinationScheme recombinationScheme = fastjet::E_scheme;
  fastjet::Strategy            strategy            = fastjet::Best;
  fastjet::JetDefinition theJetDefinition = fastjet::JetDefinition(fastjet::antikt_algorithm,
						    R,
						    recombinationScheme,
						    strategy);
    /*
   * electron isolation radius
   */
  double electronIsoR = 0.2;
  /*
   * muon isolation radius
   */
  double muonIsoR = 0.2;
  /*
   * electron isolation fraction
   */
  double electronIsoFrac = 0.1;  
  /*
   * muon isolation fraction
   */
  double muonIsoFrac = 0.1;  
  /*
   * Apply electron isolation criteria or not?
   */
  bool electronisolation = true;
  /*
   * Apply muon isolation criteria or not?
   */
  bool muonisolation = true;

 
  /* 
   * COUNTERS FOR NUMBER OF EVENTS THAT PASS CUTS
   */
  double passcuts(0); //passed all cuts
  
  /* 
   * PARAMETERS AND SWITCHES
   */
 
  /* 
   * HISTOGRAMS DEFINED HERE 
   */
  TopHist h_dummy(10,output,"dummy histo", 0,1);
  TopHist h_pT_jets(50,output,"pT of jets",400, 600);
  TopHist h_y_jets(50,output,"y of jets",-2.1,2.1);


  /*
   *
   * LOOP OVER EVENTS
   * AND
   * PERFORM ANALYSIS
   *
   */
  bool perform_analysis_on_event = false;
  for(int ii = minevents; ii < maxevents; ii++) {
    
    /* IF LEVEL 3 ANALYSIS THEN
     * CHECK IF EVENT IS IN .evp FILE
     */ 
    perform_analysis_on_event = false;
    if(basic == false) { 
       for(int pp = 0; pp < npassed_previous; pp++) { if(ii == passed_event[pp]) { perform_analysis_on_event = true; } }
    }
    if(!perform_analysis_on_event && basic == false) { continue; }

    /* GRAB EVENT ENTRY
     * FROM ROOT FILE
     * AND PRINT EVENT NUMBER
     */
    t.GetEntry(ii);

    if(ii%1 == 0) { cout << "Event number: " << ii << "\r" << flush; }

    /* 
     * Print optional weights for debugging
     */
    /*for(size_t ww = 0; ww < theOptWeightsNames->size(); ww++) {
      std::string name = (*theOptWeightsNames)[ww];
      std::cout << "theOptWeightsNames = " << name.c_str() << ":\t " << (*theOptWeights)[ww] << endl;
      }*/

    
     fastjet::PseudoJet ETmiss = fastjet::PseudoJet(theETmiss[1], theETmiss[2], theETmiss[3], theETmiss[0]);

     /* loop over particles in event
      * decide which are "photons" and "leptons", 
      * according to isolation criteria. 
      * these are then excluded from the jet finder. 
      */
     vector<fastjet::PseudoJet> photon; //all photons passing cuts
     vector<fastjet::PseudoJet> lepton, lepton_pt_unsorted, lepton_pt; //all leptons and the leptons passing cuts
     vector<int> lepton_id; //the lepton ids
     fastjet::PseudoJet pmiss; //vectors of bs and Ws


     /* create pseudojets for outgoing partons */
     vector<fastjet::PseudoJet> outgoing_partons; //all outgoing partons
     for(int ppa = 0; ppa < numoutgoing; ppa++) {
       pcurr = fastjet::PseudoJet(partons[1][ppa], partons[2][ppa], partons[3][ppa], partons[0][ppa]);
       pcurr.set_user_index(int(partons[4][ppa]));
       outgoing_partons.push_back(pcurr);
     }

     
     bool exclude_particle(false);
     for(int pp = 0; pp < numparticles; pp++) {
    
       //put the current particle into a pseudojet vector
       pcurr = fastjet::PseudoJet(objects[1][pp], objects[2][pp], objects[3][pp], objects[0][pp]);
       pcurr.set_user_index(int(objects[4][pp]));
   
       // if the particle is included in the array above, flag as excluded from particles to run over jet-finding
       for(int idl = 0; idl < exclude_ids.size(); idl++) {
	 if(fabs(objects[4][pp]) == exclude_ids[idl]) { exclude_particle = true; }
       }
            
       /* FIND HARD LEPTONS 
	* if they do not satisfy the cuts, put them into the jet finder. 
	*/
       fastjet::PseudoJet pcurr_charged; //the current particle under consideration
       //loop over particles
       if(fabs(objects[4][pp]) == 11 && pcurr.perp() > cut_pt_electron && fabs(pcurr.eta()) < cut_eta_electron) {
	 if(!electronisolation) {
	   lepton.push_back(pcurr); 
	   lepton_id.push_back(int(objects[4][pp]));
	   exclude_particle = true;
	 } else if(electronisolation) {
	   bool iso_electron = 1;
	   double sum_pt_tracks = 0;
	   for(int yy = 0; yy < numparticles; yy++) {
	     if(objects[5][yy]!=0 && yy!=pp) {//if charged particle
	       pcurr_charged = fastjet::PseudoJet(objects[1][yy], objects[2][yy], objects[3][yy], objects[0][yy]);
	       if(deltaR(pcurr, pcurr_charged) < electronIsoR)
		 sum_pt_tracks += pcurr_charged.perp();
	     }
	   }
	   if(sum_pt_tracks/pcurr.perp() < electronIsoFrac) {
	     //cout << "electron isolation = " << sum_pt_tracks/pcurr.perp()  << endl;	
	     iso_electron = 1;
	     lepton.push_back(pcurr); 
	     lepton_id.push_back(int(objects[4][pp]));
	     exclude_particle = true;
	   }	  
	 }
       }
	
       if(fabs(objects[4][pp]) == 13 && pcurr.perp() > cut_pt_muon && fabs(pcurr.eta()) < cut_eta_muon) {
	 if(!muonisolation) {
	   lepton.push_back(pcurr); 
	   lepton_id.push_back(int(objects[4][pp]));
	   exclude_particle = true;
	 } else if(muonisolation) {
	   bool iso_muon = 1;
	   double sum_pt_tracks = 0;
	   for(int yy = 0; yy < numparticles; yy++) {
	     if(objects[5][yy]!=0&&yy!=pp) {//if charged particle
	       pcurr_charged = fastjet::PseudoJet(objects[1][yy], objects[2][yy], objects[3][yy], objects[0][yy]);
	       if(deltaR(pcurr, pcurr_charged) < muonIsoR)
		 sum_pt_tracks += pcurr_charged.perp();
	     }
	   }
	   if(sum_pt_tracks/pcurr.perp() < muonIsoFrac) {
	     //cout << "muon isolation = " << sum_pt_tracks/pcurr.perp()  << endl;	
	     iso_muon = 1;
	     lepton.push_back(pcurr); 
	     lepton_id.push_back(int(objects[4][pp]));
	     exclude_particle = true;
	   }	  
	 }
       }
    
       if( (fabs(objects[4][pp]) == 22 && pcurr.perp() > cut_pt_photon && fabs(pcurr.eta()) < cut_eta_photon) ) { 
	 photon.push_back(pcurr); 
	 exclude_particle = true; 
       }



       // calculate missing energy 4-vector (using all visible objects)  
    if(fabs(pcurr.eta()) < cut_eta && pcurr.perp() > cut_pt_part && fabs(objects[4][pp]) != 12 && fabs(objects[4][pp]) != 14 && fabs(objects[4][pp]) != 16) {
      pmiss -= pcurr;
    }//end of calculate pmiss;
    
    //push the visible particles into the jet finding input and calculate the missing energy 4-vector
    if(fabs(pcurr.eta()) < cut_eta && pcurr.perp() > cut_pt_part && !exclude_particle) {
      input_particles.push_back(pcurr);
    }
    
    //reset the exclude_particle flag
    exclude_particle = false;	    
  } /* LOOP OVER PARTICLES OF EVENT
     * ENDS
     * HERE 
     */

     /* 
      * Jet finding here:
      */
      
     fastjet::ClusterSequence clust_seq(input_particles, theJetDefinition);
     vector<fastjet::PseudoJet> jets = sorted_by_pt(clust_seq.inclusive_jets(cut_pt_jet));


     /* at this point, "jet" is a vector of PseudoJets that contains all the jets, 
      * lepton contains all the isolated (or not, depending on choice) charged leptons 
      * photon contains all the isolated photons. 
      */ 
     

    /*
     * FILL IN THE HISTOGRAMS
     */
     // loop over partons:
     for(int jp = 0; jp < outgoing_partons.size(); jp++) {
       cout << outgoing_partons[jp] << "\t" << outgoing_partons[jp].user_index() << endl;
     }
     //loop over jets:
     cout << "jets:" << endl;
     for(int js = 0; js < jets.size(); js++) {
       cout << jets[js] << endl;
       if(jets[js].perp() > 500. && jets[js].perp() < 550. && fabs(jets[js].rapidity()) < 1.7) {
	 h_pT_jets.thfill(jets[js].perp(), evweight);
	 h_y_jets.thfill(jets[js].rapidity(), evweight);
       }
     }

    /* 
     * APPLY CUTS 
     */

    /*
     * DOES THE EVENT PASS ALL THE CUTS?
     * IF SO INCREMENT THE WEIGHT
     */ 
    passcuts+=evweight;

    /*
     * Fill in the _var.root file for further analysis.
     */
    
    //    variables[0] = ;  
    eventweight[0] = evweight;

    Data2->Fill();
    
    /* IF EVENT HAS PASSED CUTS
     * PRINT TO .evp or .evp2 FILE 
     * INCREMENT AND CONTINUE
     */
    outevp << ii << endl;

    /*
     * Clear vectors from memory
     */
    input_particles.clear();
    photon.clear();
    lepton.clear();
    jets.clear();
    
    
		    
  } /* LOOP OVER EVENTS ENDS HERE 
     * ENDS HERE
     */
  Data2->GetCurrentFile();
  Data2->Write();
  dat2->Close();
  cout << "A root tree has been written to the file: " << fnameroot << endl;

  /* OUTPUT HISTOGRAMS
   * HERE AND 
   * FINISH
   */
  h_dummy.plot(output,1,0);
  h_pT_jets.add(output,1,0);
  h_y_jets.add(output,1,0);

  
  cout << "------------------" << endl;
  cout << "passed = " << passcuts << endl;
  cout << "------------------" << endl;

  return 0;
}

double dot(fastjet::PseudoJet p1, fastjet::PseudoJet p2) {
  return (p1.e() * p2.e() - p1.px() * p2.px() - p1.py() * p2.py() - p1.pz() * p2.pz() );
}


double deltaR(fastjet::PseudoJet p1, fastjet::PseudoJet p2) { 
  double dphi_tmp; 

  dphi_tmp = p2.phi() - p1.phi();
  if(dphi_tmp > M_PI) 
    dphi_tmp = 2 * M_PI - dphi_tmp;
  else if( dphi_tmp < - M_PI)  
    dphi_tmp = 2 * M_PI + dphi_tmp;
  
  //  return sqrt(sqr(p1.eta() - p2.eta()) + sqr(dphi_tmp));
  return sqrt(sqr(p1.rap() - p2.rap()) + sqr(dphi_tmp));
}

//----------------------------------------------------------------------
// does the actual work for printing out a jet
//----------------------------------------------------------------------
ostream & operator<<(ostream & ostr, const PseudoJet & jet) {
  ostr << "e, pt, y, phi =" 
       << " " << setw(10) <<  jet.e()  
       << " " << setw(10) << jet.perp() 
       << " " << setw(6) <<  jet.rap()  
       << " " << setw(6) <<  jet.phi()  
       << ", mass = " << setw(10) << jet.m()
       << ", btag = " << jet.user_index();
  return ostr;
}
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}


bool btag_hadrons(fastjet::PseudoJet jet) {
  bool btagged(false);
  /* search constintuents of jets for b-mesons */
  for(int cc = 0; cc < jet.constituents().size(); cc++) { 
    for(int bb = 0; bb < 105; bb++) { 
      if(jet.constituents()[cc].user_index() == bhadronid[bb]) { 
	btagged = true;
	//	cout << "Jet B-tagged!" << endl;
	//	cout << jet << endl;
      }
    }
  }
  return btagged;
}
	

fastjet::PseudoJet smear_jet(fastjet::PseudoJet jet_in) {
  if(donotsmear_jets) { return jet_in; }

  fastjet::PseudoJet smeared; 
  double smearing = 20, smeared_pt(0);

  double pt = jet_in.perp();
  double eta = fabs(jet_in.eta());
  double sigma(0);
  
  double a, b, S, C;
  if(eta < 0.8) { a = 3.2; b = 0.07; S = 0.74; C = 0.05; }
  if(eta > 0.8 && eta < 1.2) { a = 3.0; b = 0.07; S = 0.81; C = 0.05; }
  if(eta > 1.2 && eta < 2.8) { a = 3.3; b = 0.08; S = 0.54; C = 0.05; }
  if(eta > 2.8 /*&& eta < 3.6*/) { a = 2.8; b = 0.11; S = 0.83; C = 0.05; }

  double mu_pileup = 40;
  double N = a + b * mu_pileup;

  sigma = pt * sqrt( sqr(N)/sqr(pt) + sqr(S) / pt + sqr(C) );

  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;
  
  double deltaE = - jet_in.e() + sqrt( sqr(jet_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*jet_in.px() + smeared_pt*sin(theta)*sin(phi)*jet_in.py() + smeared_pt*cos(theta)*jet_in.pz()));

  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);

  
  smeared = jet_in + smearing_vector;

  return smeared;
}

fastjet::PseudoJet smear_photon(fastjet::PseudoJet photon_in) {
  if(donotsmear_photons) { return photon_in; }

  fastjet::PseudoJet smeared;
  double smeared_pt = 0;
  double smear_frac = 0.1E-2;
  double smear_sampling = 0.15;
  double sigma(smear_sampling * sqrt(photon_in.perp()) + smear_frac*photon_in.perp());

  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;

  double deltaE = - photon_in.e() + sqrt( sqr(photon_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*photon_in.px() + smeared_pt*sin(theta)*sin(phi)*photon_in.py() + smeared_pt*cos(theta)*photon_in.pz()));
  
  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);
  smeared = photon_in + smearing_vector;
  //cout << "smeared mass = " << smeared.m() << endl;
  return smeared;

}



fastjet::PseudoJet smear_lepton(fastjet::PseudoJet lepton_in, int lepton_id) {

  if(donotsmear_leptons) { return lepton_in; }
   
    
  fastjet::PseudoJet smeared;
  double smeared_pt = 0;
  double smearing = 20.;

  double pt = lepton_in.perp();
  double lepton_energy = lepton_in.e();
  double eta = fabs(lepton_in.eta());
  double sigma(0);

  //see ATL-PHYS-PUB-2013-009
  if(lepton_id == 13) {
    double sigma_id = 0;
    double sigma_ms = 0;
    double sigma_cb = 0;
    double a1, a2, b0, b1, b2;
    
    if(eta < 0.18) { a1 = 0.01061; a2 = 0.000157; }
    if(eta > 0.18 && eta < 0.36) { a1 = 0.01084; a2 = 0.000153; }
    if(eta > 0.36 && eta < 0.54) { a1 = 0.01124; a2 = 0.000150; }
    if(eta > 0.54 && eta < 0.72) { a1 = 0.01173; a2 = 0.000149; }
    if(eta > 0.72 && eta < 0.90) { a1 = 0.01269; a2 = 0.000148; }
    if(eta > 0.90 && eta < 1.08) { a1 = 0.01406; a2 = 0.000161; }
    if(eta > 1.08 && eta < 1.26) { a1 = 0.01623; a2 = 0.000192; }
    if(eta > 1.26 && eta < 1.44) { a1 = 0.01755; a2 = 0.000199; } 
    if(eta > 1.44 && eta < 1.62) { a1 = 0.01997; a2 = 0.000232; } 
    if(eta > 1.62 && eta < 1.80) { a1 = 0.02453; a2 = 0.000261; }
    if(eta > 1.80 && eta < 1.98) { a1 = 0.03121; a2 = 0.000297; }
    if(eta > 1.98 && eta < 2.16) { a1 = 0.03858; a2 = 0.000375; }
    if(eta > 2.16 && eta < 2.34) { a1 = 0.05273; a2 = 0.000465; }
    if(eta > 2.34 && eta < 2.52) { a1 = 0.05329; a2 = 0.000642; }
    if(eta > 2.52 /*&& eta < 2.70*/) { a1 = 0.05683; a2 = 0.000746; }

    if(eta < 1.05) { b1 = 0.02676; b2 = 0.00012; }
    if(eta > 1.05) { b1 = 0.03880; b2 = 0.00016; }

    sigma_id = pt * sqrt( a1 + sqr(a2 * pt) );
    sigma_ms = pt * sqrt( sqr(b0/pt) + sqr(b1) + sqr(b2*pt) );
    sigma = (sigma_id * sigma_ms)/sqrt( sqr(sigma_id) + sqr(sigma_ms) ); //sigma_cb

  }


  if(lepton_id == 11) {
    double sigma = 0;
    if(eta < 1.4) { sigma = sqrt( sqr(0.3) + sqr(0.10 * sqrt(lepton_energy)) + sqr( 0.010 * lepton_energy ) ); }
    if(eta > 1.4 /* && eta < 2.47 */) { sigma = sqrt( sqr(0.3) + sqr(0.15 * sqrt(lepton_energy)) + sqr( 0.015 * lepton_energy ) ); }
  }

  smeared_pt = fabs(rnd.Gaus(0,sigma));
  double theta = rnd.Rndm()*M_PI;
  double phi = rnd.Rndm()*2.*M_PI;
  
  
  double deltaE = - lepton_in.e() + sqrt( sqr(lepton_in.e()) + sqr(smeared_pt) + 2 * (smeared_pt*sin(theta)*cos(phi)*lepton_in.px() + smeared_pt*sin(theta)*sin(phi)*lepton_in.py() + smeared_pt*cos(theta)*lepton_in.pz()));
  
  fastjet::PseudoJet smearing_vector(smeared_pt*sin(theta)*cos(phi),smeared_pt*sin(theta)*sin(phi), smeared_pt*cos(theta), deltaE);
  
  smeared = lepton_in + smearing_vector;  
  
  //smeared.reset(smeared.px(), smeared.py(), smeared.pz(), eprime);
  
  return smeared;
}

bool lepton_efficiency_accept(fastjet::PseudoJet lepton_in, int lepton_id) {
  bool accepted(1);
  if(donot_apply_efficiency) { return accepted; }

  double pt = lepton_in.perp();
  double eta = fabs(lepton_in.eta());
  
  double epsilon = 0;
  if(lepton_id == 11) {
    epsilon = 0.85 - 0.191 * exp(1 - pt/20);
  }
  if(lepton_id == 13) {
    if(eta<0.1) { epsilon = 0.54; }
    if(eta>0.1) { epsilon = 0.97; } 
  }
  double random_num = rnd.Rndm();
  //  cout << lepton_id << " " << pt << " " << eta << " " << random_num << " " << epsilon << endl;
  if(random_num > epsilon) { accepted = 0; }
  return accepted;
}
bool photon_efficiency_accept(fastjet::PseudoJet photon_in) {
  bool accepted(1);
  if(donot_apply_efficiency) { return accepted; }

  double pt = photon_in.perp();
  double eta = fabs(photon_in.eta());
  
  double epsilon = 0;
  
  epsilon = 0.76 - 1.98 * exp(-pt/16.1);
 
  double random_num = rnd.Rndm();
  if(random_num > epsilon) { accepted = 0; }
  return accepted;
}
bool jet_efficiency_accept(fastjet::PseudoJet jet_in) {
    bool accepted(1);
    if(donot_apply_efficiency) { return accepted; }

    double pt = jet_in.perp();
    double epsilon = 0;

    epsilon = 0.75 + (0.95 - 0.75) * pt / (50. - 20.);
    if(epsilon < 0) { epsilon = 0; }
    if(epsilon > 1.0) { epsilon = 1.0; }

    
    double random_num = rnd.Rndm();
    if(random_num > epsilon) { accepted = 0; }
    return accepted;
    
}

                     
