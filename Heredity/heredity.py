import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    
    def parent_prob(parent):
        if parent in one_gene:
            parent_pass = 0.5
        elif parent in two_genes:
            parent_pass = 1 - PROBS["mutation"]
        else:
            parent_pass = PROBS["mutation"]
    
        return parent_pass
    
    joint = 1
    
    for person in people:
        if person in one_gene:
            person_gene = 1
        elif person in two_genes:
            person_gene = 2
        else:
            person_gene = 0
        
        
        person_trait = person in have_trait
        
        mother = people[person]["mother"]
        father = people[person]["father"]
        
        if mother is None: # might need to be changed
            person_prob = PROBS["gene"][person_gene]
        else:
            
            mother_pass = parent_prob(mother)
            father_pass = parent_prob(father)
            
            
            if person_gene == 0:
                person_prob = (1-mother_pass) * (1-father_pass)
            elif person_gene == 1:
                person_prob = mother_pass*(1-father_pass) + (1-mother_pass)*father_pass
            else:
                person_prob = mother_pass * father_pass
        
        trait_prob = PROBS["trait"][person_gene][person_trait]
        joint *= person_prob * trait_prob
        
    return joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    
    for person in probabilities:
        if person in one_gene:
            person_gene = 1
        elif person in two_genes:
            person_gene = 2
        else:
            person_gene = 0
        
        if person in have_trait:
            person_trait = True
        else:
            person_trait = False
            
        probabilities[person]["gene"][person_gene] += p
        probabilities[person]["trait"][person_trait] += p
    


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    
   
    true_false = (True, False)
    
    # Get normalization dividers
    for person in probabilities:
        sum_gene = 0
        sum_trait = 0
        
        for gene in probabilities[person]["gene"]:
            sum_gene += probabilities[person]["gene"][gene] 
            
        for trait in probabilities[person]["trait"]:
                sum_trait += probabilities[person]["trait"][trait]
        
        # Normalize values
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= sum_gene
            
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= sum_trait



if __name__ == "__main__":
    main()
