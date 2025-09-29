

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(on c a)
(on d e)
(ontable e)
(ontable f)
(clear c)
(clear d)
(clear f)
)
(:goal
(and
(on a f)
(on c a)
(on f b))
)
)


