//! The domain table places every `PropertyId` exactly once, or names it in the exclusion list.

use periodica_mat::domains::{domain_of, DOMAINS, EXCLUDED};
use periodica_mat::PropertyId;

#[test]
fn every_property_id_is_placed_exactly_once() {
    for &id in PropertyId::ALL {
        let in_domains = DOMAINS.iter().filter(|d| d.ids.contains(&id)).count();
        let excluded = EXCLUDED.contains(&id);
        assert!(
            in_domains + usize::from(excluded) == 1,
            "{id:?}: in {in_domains} domains, excluded={excluded}; place it in domains.rs"
        );
    }
}

#[test]
fn no_domain_repeats_an_id_and_acoustic_is_an_empty_gap() {
    for d in DOMAINS {
        for (i, id) in d.ids.iter().enumerate() {
            assert!(!d.ids[..i].contains(id), "{id:?} repeated in {}", d.name);
        }
    }
    let acoustic = DOMAINS.iter().find(|d| d.name == "acoustic");
    assert!(acoustic.is_some_and(|d| d.ids.is_empty()));
    assert_eq!(domain_of(PropertyId::Density), Some("mass"));
    assert_eq!(domain_of(PropertyId::AtomicMass), None);
}
