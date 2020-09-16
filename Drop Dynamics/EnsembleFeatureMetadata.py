class PatchDesc(NamedTuple):
    memberName: str
    coords: Tuple[int, ...]

class FeatureRange(NamedTuple):
    start: int
    stop: Optional[int]
    name: str


class EnsembleFeatureMetadata:
    """
    Feature metadata describes manually-labeled feature regions that can be found
    in the ensemble data.
    """

    class Member:

        def __init__(self, name: Optional[str] = None):
            self.name = name  # type: Optional[str]
            self.defaultFeature = None  # type: Optional[str]
            self.features = []  # type: List[FeatureRange]

    def __init__(self):
        self.featuresPerMember = {}  # type: Dict[str, EnsembleFeatureMetadata.Member]
        self.memberNames = []  # type: List[str]
        self.memberShapes = {}  # type: Dict[str, Tuple[int, ...]]

    def build_metadata_subset(self, removedMembers: List[str], allowMissing: bool = False):
        """
        Build a new metadata object by removing some of the members.
        Useful for ignoring members present in the query when computing coverage metrics.
        """

        metadataSubset = copy.deepcopy(self)
        for memberName in removedMembers:
            if allowMissing and memberName not in metadataSubset.featuresPerMember:
                continue

            del metadataSubset.featuresPerMember[memberName]
            del metadataSubset.memberShapes[memberName]
            metadataSubset.memberNames.remove(memberName)

        return metadataSubset

    @classmethod
    def load_from_json(cls, filepath: str):
        with open(filepath, 'r') as file:
            dataJson = json.load(file)
            metadata = cls()
            metadata.memberNames = []
            metadata.memberShapes = {}

            for name, memberJson in dataJson.items():
                member = EnsembleFeatureMetadata.Member()
                member.name = name
                member.defaultFeature = memberJson['default'] if 'default' in memberJson else None
                for featureJson in memberJson['features']:
                    start, stop = featureJson['range'][0], featureJson['range'][1]
                    featureName = featureJson['name']
                    member.features.append(FeatureRange(start, stop, featureName))

                metadata.featuresPerMember[name] = member
                metadata.memberNames.append(member.name)
                metadata.memberShapes[member.name] = (tuple(int(x) for x in memberJson['shape']))

        return metadata

    def write_to_json(self, filepath: str):
        file_tools.create_dir(os.path.dirname(filepath))

        dataJson = {}
        for name, member in self.featuresPerMember.items():
            memberJson = {}
            memberJson['shape'] = self.memberShapes[name]
            memberJson['default'] = member.defaultFeature
            memberJson['features'] = []
            for feature in member.features:
                featureJson = {
                    'range': [feature[0], feature[1]],
                    'name': feature[2]
                }
                memberJson['features'].append(featureJson)

            dataJson[name] = memberJson

        with open(filepath, 'w') as file:
            json.dump(dataJson, file)

    def get_patch_features(self, patch: PatchDesc, patchShape: Tuple[int, ...]) -> Set[str]:
        result = set()

        memberMeta = self.featuresPerMember[patch.memberName]
        for feature in memberMeta.features:
            start, stop = patch.coords[0], patch.coords[0] + patchShape[0]
            # Check interval overlap.
            rangeStart, rangeStop = feature[0], feature[1] or float('inf')  # Range can specify 'null' to mean 'to the end'.
            if start <= rangeStop and rangeStart <= stop:
                result.add(feature[2])
                # If the patch protrudes outside the region - we also 'see' the default feature around it.
                if memberMeta.defaultFeature and (start < rangeStart or stop > rangeStop):
                    result.add(memberMeta.defaultFeature)

        # If none of the regions were hit - we hit the default feature outside the regions.
        if memberMeta.defaultFeature and len(result) == 0:
            result.add(memberMeta.defaultFeature)

        return result

    def get_feature_masks(self, feature):
        featureMasks = {}
        for name, memberMeta in self.featuresPerMember.items():
            memberDuration = self.memberShapes[name][0]

            featureMask = np.zeros((memberDuration,), dtype=np.bool)
            defaultMask = np.ones((memberDuration,), dtype=np.bool)

            for featureDesc in memberMeta.features:
                # Range can specify 'null' to mean 'to the end'.
                rangeStart, rangeStop = featureDesc[0], featureDesc[1] or memberDuration
                if feature == featureDesc.name:
                    # Don't overwrite with false if feature is different, we allow multiple features per frame.
                    featureMask[rangeStart:rangeStop] = True

                # Mark it in the default mask though, so we don't set it to default when another feature is present.
                defaultMask[rangeStart:rangeStop] = False

            if feature == memberMeta.defaultFeature:
                featureMask[defaultMask] = True

            featureMasks[name] = featureMask

        return featureMasks

    def get_patches_to_cover(self,
                             features: Set[str],
                             patchShape: Tuple[int, ...],
                             patches: List[PatchDesc]) -> Tuple[int, int]:
        """
        Compute the exact number of patches needed to completely cover the features.
        """
        patchesToCoverExact = 0
        patchesToCoverPartial = 0

        for patch in patches:
            patchFeatures = self.get_patch_features(patch, patchShape)
            if patchFeatures == features:
                patchesToCoverExact += 1
            if len(patchFeatures & features) > 0:
                patchesToCoverPartial += 1

        return patchesToCoverExact, patchesToCoverPartial

    def get_total_feature_coverage(self,
                                   features: Set[str],
                                   patchShape: Tuple[int, ...],
                                   patchesSorted: List[PatchDesc],
                                   isExactNotPartial: bool = False) -> Tuple[float, int]:

        masksPerFeature = []  # A (dict of masks for each member) for each feature.
        for feature in features:
            masksPerFeature.append(self.get_feature_masks(feature))

        # Aggregate the feature masks based on the metric type.
        # Take all the masks for each member and aggregate them.
        # A bit messy because we have a list of dicts, not a dict of lists.
        aggFunc = np.logical_and if isExactNotPartial else np.logical_or
        featureMasksPerMember = {memberName: functools.reduce(aggFunc, [d[memberName] for d in masksPerFeature])
                                 for memberName in masksPerFeature[0].keys()}

        totalFeatureDuration = sum((np.count_nonzero(mask) for mask in featureMasksPerMember.values()))

        patchesToCoverExact, patchesToCoverPartial = self.get_patches_to_cover(set(features), patchShape, patchesSorted)
        patchesToCover = patchesToCoverExact if isExactNotPartial else patchesToCoverPartial

        totalPatchDuration = 0
        totalFeatureCoverage = 0
        for name, memberMeta in self.featuresPerMember.items():
            memberDuration = self.memberShapes[name][0]

            featureMask = featureMasksPerMember[name]
            patchMask = np.zeros((memberDuration,), dtype=np.bool)

            for patch in patchesSorted[:patchesToCover]:
                if patch.memberName != name:
                    continue

                patchMask[patch.coords[0]:patch.coords[0]+patchShape[0]] = True

            coverageMask = np.logical_and(featureMask, patchMask)

            # Draw diagrams showing temporal feature coverage.
            # print("--- Member {}".format(name))
            # print("Feat. {}".format(''.join('X' if v else '-' for v in featureMask)))
            # print("Patch {}".format(''.join('X' if v else '-' for v in patchMask)))
            # print("Slots {}".format(''.join('^' if v % patchShape[0] == 0 else '-' for v in range(memberDuration))))

            totalPatchDuration += np.count_nonzero(patchMask)
            totalFeatureCoverage += np.count_nonzero(coverageMask)

        # print("Feat. {} Patch. {} Cov. {}".format(totalFeatureDuration, totalPatchDuration, totalFeatureCoverage))

        return totalFeatureCoverage / totalFeatureDuration * 100, patchesToCover

    def apply_coord_mapping(self, axisMaps: List[CoordMap]):
        """
        Since we crop and downsample the original data, the coordinates specified in the feature metadata need
        to be mapped accordingly.
        """

        # We only map the time axis here.
        left, right, scale = axisMaps[0]

        # Map the feature descriptions.
        for name, memberMeta in self.featuresPerMember.items():
            def _map_feature(feature: Tuple[int, int, str]):
                start, stop, featureName = feature
                start = int(max(0, start - left) / scale)
                stop = int(max(0, min(stop or right, right) - left) / scale)  # 'Stop' might be none.

                return FeatureRange(start, stop, featureName)

            memberMeta.features = list(map(_map_feature, memberMeta.features))

        # Map the member shapes.
        for name, shape in self.memberShapes.items():
            self.memberShapes[name] = tuple(int((min(x, m.right) - max(0, m.left)) / m.scale)
                                            for x, m in zip(shape, axisMaps))
